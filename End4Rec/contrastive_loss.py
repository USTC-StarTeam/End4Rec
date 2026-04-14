import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
### 
class ContrastiveLossCell(nn.Cell):
    def __init__(self):
        super(ContrastiveLossCell, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.reduce_mean = ops.ReduceMean()
        self.log = ops.Log()
    def construct(self, S_clean, S_orig, S_noise):
        Q_clean = ops.ReduceMean(keep_dims=False)(S_clean, 1)
        Q_orig = ops.ReduceMean(keep_dims=False)(S_orig, 1)
        Q_noise = ops.ReduceMean(keep_dims=False)(S_noise, 1)
        diff1 = Q_clean - Q_orig
        diff2 = Q_orig - Q_noise
        loss1 = -self.reduce_mean(self.log(self.sigmoid(diff1)))
        loss2 = -self.reduce_mean(self.log(self.sigmoid(diff2)))
        return loss1 + loss2
