import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter

class BehaviorAwareEmbedding(nn.Cell):
    def __init__(self, num_items, num_behaviors, seq_length, d_model):
        super(BehaviorAwareEmbedding, self).__init__()
        self.item_emb = nn.Embedding(num_items, d_model)
        self.beh_emb = nn.Embedding(num_behaviors, d_model)
        self.pos_emb = nn.Embedding(seq_length, d_model)
    def construct(self, item_ids, behavior_ids, positions):
        return self.item_emb(item_ids) + self.beh_emb(behavior_ids) + self.pos_emb(positions)

class BlockMLP(nn.Cell):
    def __init__(self, block_size):
        super(BlockMLP, self).__init__()
        self.fc1 = nn.Dense(block_size, block_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(block_size, block_size)
    def construct(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EBMEnhanced(nn.Cell):
    def __init__(self, d_model, seq_length, num_blocks=4, epsilon=1e-3):
        super(EBMEnhanced, self).__init__()
        assert d_model % num_blocks == 0
        self.d_model = d_model
        self.seq_length = seq_length
        self.num_blocks = num_blocks
        self.block_size = d_model // num_blocks
        self.epsilon = epsilon
        self.block_mlps = nn.CellList([BlockMLP(self.block_size) for _ in range(num_blocks)])
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=2)
        self.batch_matmul = ops.BatchMatMul()
        self.transpose = ops.Transpose()
        self.eye = ops.Eye()
        self.tile = ops.Tile()
        self.reduce_mean = ops.ReduceMean()
        self.log = ops.Log()
        self.cholesky = ops.Cholesky()
        self.diagpart = ops.DiagPart()
    def _compute_logdet(self, mat):
        chol = self.cholesky(mat)
        diag = self.diagpart(chol)
        return 2 * ops.ReduceSum()(self.log(diag), -1)
    
    
    def construct(self, x):
        B, L, d = x.shape
        x_np = x.asnumpy()
        X_np = np.fft.fft(x_np, axis=1)
        X = Tensor(X_np, ms.complex64)
        X_blocks = self.reshape(X, (B, L, self.num_blocks, self.block_size))
        fused_blocks = []
        for i in range(self.num_blocks):
            X_i = X_blocks[:, :, i, :]
            X_i_real = ops.Real()(X_i)
            X_i_fused = self.block_mlps[i](X_i_real)
            X_i_complex = ops.Cast()(X_i_fused, ms.complex64)
            fused_blocks.append(X_i_complex)
        X_fused = self.concat(fused_blocks)
        X_fused_real = ops.Real()(X_fused)
        gram = self.batch_matmul(X_fused_real, self.transpose(X_fused_real, (0,2,1)))
        I = self.eye(L, L, ms.float32)
        I = ops.ExpandDims()(I, 0)
        I = self.tile(I, (B, 1, 1))
        coeff = self.d_model / (L * (self.epsilon ** 2))
        gram_reg = I + coeff * gram
        logdet_vals = self._compute_logdet(gram_reg)
        reg_loss = 0.5 * self.reduce_mean(logdet_vals)
        X_fused_np = np.fft.ifft(X_fused.asnumpy(), axis=1).real
        out = Tensor(X_fused_np, ms.float32)
        return out, reg_loss

class HardNoiseEliminator(nn.Cell):
    def __init__(self, d_model, temperature=1.0):
        super(HardNoiseEliminator, self).__init__()
        self.fc = nn.Dense(d_model, 1)
        self.sigmoid = ops.Sigmoid()
        self.temperature = temperature
    def construct(self, S):
        t = self.fc(S)
        p_b = ops.OnesLike()(t)
        diff = p_b - t
        prob = self.sigmoid(diff)
        logits = ops.Concat(axis=-1)((prob, 1-prob))
        mask = ops.Greater()(prob, 0.5)
        mask = ops.Cast()(mask, ms.float32)
        S_hp = S * mask
        S_hn = S * (1-mask)
        return S_hp, S_hn

class SoftNoiseFilter(nn.Cell):
    def __init__(self, d_model, num_behaviors):
        super(SoftNoiseFilter, self).__init__()
        self.num_behaviors = num_behaviors
        self.linears = nn.CellList([nn.Dense(d_model, d_model) for _ in range(num_behaviors)])
    def construct(self, S_hp, behavior_ids):
        B, L, d = S_hp.shape
        outputs_sp = []
        outputs_sn = []
        for i in range(B):
            tokens = S_hp[i]
            beh_ids = behavior_ids[i]
            sp_list = []
            for j in range(L):
                beh = int(beh_ids[j].asnumpy())
                token = tokens[j:j+1, :]
                sp_list.append(self.linears[beh](token))
            sp = ops.Concat(axis=0)(sp_list)
            sn = tokens - sp
            outputs_sp.append(sp.expand_dims(0))
            outputs_sn.append(sn.expand_dims(0))
        S_sp = ops.Concat(axis=0)(outputs_sp)
        S_sn = ops.Concat(axis=0)(outputs_sn)
        return S_sp, S_sn

class ContrastiveLossCell(nn.Cell):
    def __init__(self):
        super(ContrastiveLossCell, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.reduce_mean = ops.ReduceMean()
    def construct(self, S_clean, S_orig, S_noise):
        Q_clean = ops.ReduceMean(keep_dims=False)(S_clean, 1)
        Q_orig = ops.ReduceMean(keep_dims=False)(S_orig, 1)
        Q_noise = ops.ReduceMean(keep_dims=False)(S_noise, 1)
        diff1 = Q_clean - Q_orig
        diff2 = Q_orig - Q_noise
        loss1 = -self.reduce_mean(ops.Log()(self.sigmoid(diff1)))
        loss2 = -self.reduce_mean(ops.Log()(self.sigmoid(diff2)))
        return loss1 + loss2

class END4Rec(nn.Cell):
    def __init__(self, num_items, num_behaviors, seq_length, d_model, num_blocks=4, epsilon=1e-3):
        super(END4Rec, self).__init__()
        self.embedding = BehaviorAwareEmbedding(num_items, num_behaviors, seq_length, d_model)
        self.ebm = EBMEnhanced(d_model, seq_length, num_blocks, epsilon)
        self.hard_noise = HardNoiseEliminator(d_model)
        self.soft_noise = SoftNoiseFilter(d_model, num_behaviors)
        self.output_layer = nn.Dense(d_model, num_items)
    def construct(self, item_ids, behavior_ids, positions):
        x = self.embedding(item_ids, behavior_ids, positions)
        S, reg_loss = self.ebm(x)
        S_hp, S_hn = self.hard_noise(S)
        S_sp, S_sn = self.soft_noise(S_hp, behavior_ids)
        final_rep = S_sp[:, -1, :]
        logits = self.output_layer(final_rep)
        return logits, reg_loss, S, S_hp, S_hn, S_sp, S_sn

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class End4RecLoss(nn.Cell):
    def __init__(self, model, loss_fn, reg_weight=0.01, contrast_weight=0.1):
        super(End4RecLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.reg_weight = reg_weight
        self.contrast_weight = contrast_weight
        self.contrast_loss_fn = ContrastiveLossCell()

    def construct(self, item_ids, behavior_ids, positions, labels, neg_ids):
        logits, reg_loss, S, S_hp, S_hn, S_sp, S_sn = self.model(item_ids, behavior_ids, positions)
        pos_ids = item_ids  
        neg_ids = neg_ids  
        seq_out = logits  
        pred_loss = self.cross_entropy(seq_out, pos_ids, neg_ids)

        contrast_loss_hard = self.contrast_loss_fn(S_hp, S, S_hn)
        contrast_loss_soft = self.contrast_loss_fn(S_sp, S_hp, S_sn)
        
        total_loss = pred_loss + self.reg_weight * reg_loss + self.contrast_weight * (contrast_loss_hard + contrast_loss_soft)
        
        return total_loss

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        seq_emb = seq_out[:, -1, :]  

        pos_logits = ops.ReduceSum()(pos_emb * seq_emb, -1)  
        neg_logits = ops.ReduceSum()(neg_emb * seq_emb, -1) 

        loss = ops.ReduceMean()(
            - ops.Log()(ops.Sigmoid()(pos_logits) + 1e-24) -
            ops.Log()(1 - ops.Sigmoid()(neg_logits) + 1e-24)
        )

        return loss

