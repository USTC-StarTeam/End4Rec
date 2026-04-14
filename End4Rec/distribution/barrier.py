import mindspore as ms
import mindspore.ops as ops
from mindspore.communication.management import init, get_rank, get_group_size
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU")
init()
rank = get_rank()
group_size = get_group_size()
dummy = ms.Tensor(1.0, ms.float32)
allreduce = ops.AllReduce()
result = allreduce(dummy) / group_size
print("Barrier result:", result.asnumpy())


# import mindspore as ms
# def merge_checkpoints(ckpt_paths, save_path):
#     from mindspore.train.serialization import load_checkpoint, save_checkpoint
#     params = {}
#     for ckpt in ckpt_paths:
#         ckpt_dict = load_checkpoint(ckpt)
#         for key, value in ckpt_dict.items():
#             if key in params:
#                 params[key] += value
#             else:
#                 params[key] = value
#     group_size = ms.communication.management.get_group_size()
#     for key in params:
#         params[key] = params[key] / group_size
#     save_checkpoint(params, save_path)
