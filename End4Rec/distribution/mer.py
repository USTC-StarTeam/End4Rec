# #!/usr/bin/env python
# import argparse
# from distributed_utils import merge_checkpoints

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt_dir", type=str, default="./ckpt/")
#     parser.add_argument("--output_ckpt", type=str, default="./ckpt/merged_ckpt.ckpt")
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
#     import os
#     ckpt_paths = []
#     for file in os.listdir(args.ckpt_dir):
#         if file.endswith(".ckpt") and "rank" in file:
#             ckpt_paths.append(os.path.join(args.ckpt_dir, file))
#     if len(ckpt_paths) == 0:
#         print("No checkpoint files found!")
#     else:
#         merge_checkpoints(ckpt_paths, args.output_ckpt)
#         print("Merged checkpoint saved to:", args.output_ckpt)

# chmod +x run_distributed.sh merge_checkpoints.py
# mpirun -n 4 ./run.sh
