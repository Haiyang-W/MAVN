# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from arguments import get_args
from gibson2.utils.utils import parse_config
from env.make_collavn_env import CollaVN

import sys
import random
import json
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import os
from env.paralle_env import make_parallel_env
from tqdm import tqdm



args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def init_env(config_filename, args, data_split=None, mode='headless', render_to_tensor=True):
    choice_scene = random.choice(data_split)
    model_id = choice_scene
    return MaBase_ImgGoal_NoiseNavigateRandomEnv(config_file=config_filename, mode=mode, model_id=model_id, args=args,
                                                 action_timestep=1 / 10.0,
                                                 physics_timestep=1 / 40.0, render_to_tensor=render_to_tensor)

def main():
    basic_config = parse_config(args.task_config)
    model_dir = basic_config.get("model_path")
    torch.manual_seed(basic_config.get("seed", 0))
    np.random.seed(basic_config.get("seed", 0))
    num_scenes = args.num_processes
    num_agents = args.num_agents = len(basic_config.get('robot'))
    num_episodes = basic_config.get('episode_num')
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    gpu_id_list = list(range(torch.cuda.device_count()))
    gpu_num = len(gpu_id_list)
    num_scenes_per_gpu = num_scenes // len(gpu_id_list)
    args.num_scenes_er_gpu = num_scenes_per_gpu

    scenes_id = basic_config.get('scenes_id')

    vision_range_xy = (basic_config.get('vision_range', 64), basic_config.get('vision_range', 64))
    max_step = basic_config.get("max_episode_length")
    render_to_tensor = basic_config.get("USE_CUDA")

    data_split = random.sample(scenes_id, num_scenes)
    envs = make_parallel_env(args.task_config, args, data_split, render_to_tensor=render_to_tensor, mode='headless', gpu_id_list=gpu_id_list)

    env_gt_fp_projs = torch.zeros(num_scenes, num_agents, *vision_range_xy).float().to(device)
    env_gt_fp_explored = torch.zeros(num_scenes, num_agents, *vision_range_xy).float().to(device)
    poses_change = torch.zeros(num_scenes, num_agents, 3).float().to(device)

    for ep_num in range(num_episodes):
        all_obs, _, _, all_infos = envs.reset()  # s, a, h, w, 3
        rank_list = [s_info['rank'] for s_info in all_infos]

        for s, rk in enumerate(rank_list):
            rank = all_infos[s]['rank']
            env_gt_fp_projs[rank, :,  :, :] = all_infos[s]['multi_fp_proj'].to(device)
            env_gt_fp_explored[rank, :,  :, :] = all_infos[s]['multi_fp_explored'].to(device)
            poses_change[rank, :, :] = torch.FloatTensor(all_infos[s]['multi_sensor_pose']).to(device) # (y,x,o) means  (r,c,o)

        for step in tqdm(range(max_step)):
            actions = np.random.uniform(-1. ,1., (num_scenes, num_agents, 2))
            all_obs, rewards, dones, all_infos = envs.step(actions)

if __name__ == "__main__":
    main()