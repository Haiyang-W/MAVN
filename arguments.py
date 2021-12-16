import argparse
import math
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Active-Neural-SLAM')

    ## General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=1)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=4,
                        help="""how many training processes to use (default:4)
                                Overridden when auto_gpu_config=1
                                and training on gpus """)
    parser.add_argument('--num_processes_per_gpu', type=int, default=11)
    parser.add_argument('--num_scenes_per_gpu', type=int, default=1)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=0)
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='number of training episodes (default: 10000)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eval', type=int, default=0,
                        help='1: evaluate models (default: 0)')

    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--task_config", type=str,
                        default="",
                        help="path to config yaml containing task information")

    parser.add_argument("--split", type=str, default="tiny_train",
                        help="dataset split (train | val | val_mini) ")

    parser.add_argument('--vision_range', type=int, default=64)
    parser.add_argument('--obstacle_boundary', type=int, default=5)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=2)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help='1:Render the frame (default: 0)')
    parser.add_argument('-ot', '--obs_threshold', type=float, default=1)

    # parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "gibson" in args.task_config and \
                    "train" in args.split:
                args.total_num_scenes = 72
            elif "gibson" in args.tamap_size_cmsk_config and \
                    "val_mt" in args.split:
                args.total_num_scenes = 14
            elif "gibson" in args.task_config and \
                    "val" in args.split:
                args.total_num_scenes = 1
            else:
                assert False, "Unknown task config, please specify" + \
                        " total_num_scenes"

            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            total_num_scenes = args.total_num_scenes
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                    torch.cuda.get_device_properties(i).total_memory \
                            /1024/1024/1024)

            # num_processes_per_gpu = int(gpu_memory/2.8)
            # num_processes_on_first_gpu = int((gpu_memory - 10.0)/2.8)
            num_processes_per_gpu = int(gpu_memory/6.)
            num_processes_on_first_gpu = int((gpu_memory - 10.0)/6.)

            if num_gpus == 1:
                args.num_processes_on_first_gpu = args.total_num_scenes
                args.num_processes_per_gpu = args.total_num_scenes
                args.num_processes = args.total_num_scenes
            else:
                total_threads = num_processes_per_gpu * (num_gpus - 1) \
                                + num_processes_on_first_gpu

                num_scenes_per_thread = math.ceil(total_num_scenes / total_threads)
                num_threads = math.ceil(total_num_scenes / num_scenes_per_thread)
                args.num_processes_per_gpu = min(num_processes_per_gpu,
                                                 math.ceil(num_threads // (num_gpus - 1)))

                args.num_processes_on_first_gpu = max(0,
                                                      num_threads - args.num_processes_per_gpu * (num_gpus - 1))

                args.num_processes = num_threads

            args.sim_gpu_id = 1

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print("Number of processes on GPU 0: {}".format(
                                      args.num_processes_on_first_gpu))
            print("Number of processes per GPU: {}".format(
                                      args.num_processes_per_gpu))

    if args.num_mini_batch == "auto":
        args.num_mini_batch = args.num_processes // 2
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    return args
