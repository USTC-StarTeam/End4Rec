import argparse
import os
import time
import json
import mindspore as ms
import mindspore.context as context
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--parallel_mode", type=str, default="data_parallel")
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--config_file", type=str, default="config.json")
    return parser.parse_args()

def set_distributed_context(device_target, parallel_mode, device_num):
    context.set_context(mode=ms.context.GRAPH_MODE, device_target=device_target)
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=device_num)

def init_distributed():
    init()
    rank = get_rank()
    group_size = get_group_size()
    return rank, group_size

def distributed_barrier():
    dummy = Tensor(1.0, ms.float32)
    allreduce = ops.AllReduce()
    dummy = allreduce(dummy)
    return dummy

def map_reduce_demo():
    rank = get_rank()
    local_value = Tensor(rank + 1, ms.float32)
    allreduce = ops.AllReduce()
    total = allreduce(local_value)
    return total

def save_cluster_info(rank, group_size, args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    info = {
        "rank": rank,
        "group_size": group_size,
        "device_target": args.device_target,
        "parallel_mode": args.parallel_mode,
        "device_num": args.device_num
    }
    with open(os.path.join(args.log_dir, "cluster_info_rank{}.json".format(rank)), "w") as f:
        json.dump(info, f)

def merge_model_checkpoints(ckpt_paths, save_path):
    from mindspore.train.serialization import load_checkpoint, save_checkpoint
    params = {}
    for ckpt in ckpt_paths:
        ckpt_dict = load_checkpoint(ckpt)
        for key in ckpt_dict:
            if key in params:
                params[key] += ckpt_dict[key]
            else:
                params[key] = ckpt_dict[key]
    group_size = get_group_size()
    for key in params:
        params[key] = params[key] / group_size
    save_checkpoint(params, save_path)

def launch_cluster():
    args = parse_args()
    set_distributed_context(args.device_target, args.parallel_mode, args.device_num)
    rank, group_size = init_distributed()
    save_cluster_info(rank, group_size, args)
    print("Cluster launched: Rank {}, Group size {}".format(rank, group_size))
    barrier = distributed_barrier()
    print("Barrier result (should be 1.0):", barrier.asnumpy())
    total = map_reduce_demo()
    print("MapReduce demo, sum of ranks:", total.asnumpy())
    time.sleep(2)
    return args, rank, group_size

args, rank, group_size = launch_cluster()
if rank == 0:
    ckpt_list = []
    for r in range(group_size):
        ckpt_list.append("./ckpt/ckpt_stage4_rank{}.ckpt".format(r))
    merge_model_checkpoints(ckpt_list, "./ckpt/merged_ckpt.ckpt")
    print("Model checkpoints merged to ./ckpt/merged_ckpt.ckpt")
