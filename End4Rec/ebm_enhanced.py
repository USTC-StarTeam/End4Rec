import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from block_mlp import BlockMLP
import mindspore.nn as nn
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
        I = self.tile(I, (B,1,1))
        coeff = self.d_model / (L * (self.epsilon ** 2))
        gram_reg = I + coeff * gram
        logdet_vals = self._compute_logdet(gram_reg)
        reg_loss = 0.5 * self.reduce_mean(logdet_vals)
        X_fused_np = np.fft.ifft(X_fused.asnumpy(), axis=1).real
        out = Tensor(X_fused_np, ms.float32)
        return out, reg_loss
