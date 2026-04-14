import mindspore as ms
import mindspore.ops as ops
from mindspore.communication import get_group_size
def allreduce_tensor(tensor):
    allreduce = ops.AllReduce()
    return allreduce(tensor)
def merge_checkpoints(ckpt_paths, save_path):
    from mindspore.train.serialization import load_checkpoint, save_checkpoint
    params = {}
    for ckpt in ckpt_paths:
        p = load_checkpoint(ckpt)
        for key in p:
            if key in params:
                params[key] += p[key]
            else:
                params[key] = p[key]
    group_size = get_group_size()
    for key in params:
        params[key] = params[key] / group_size
    save_checkpoint(params, save_path)
