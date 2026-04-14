import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
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
