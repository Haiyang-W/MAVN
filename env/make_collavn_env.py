from env.mabase_env import MaBase_ImgGoal_NavigateEnv
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from transforms3d.quaternions import quat2mat, qmult
import numpy as np
import random
import pybullet as p
import cv2
import cv2 as cv
import logging
from gibson2.utils.utils import parse_config
import json
import pickle
import skimage.morphology
from env.utils.map_builder import MapBuilder
import env.utils.pose as pu
import math
import torch

class CollaVN(MaBase_ImgGoal_NavigateEnv):
    def __init__(
            self,
            config_file,
            model_id=None,
            args=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            random_height=False,
            device_idx=0,
            render_to_tensor=False,
            rank=0,
            seed=0,
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param random_height: whether to randomize height for target position (for reaching task)
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(CollaVN, self).__init__(config_file,
                                                model_id=model_id,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx,
                                                render_to_tensor=render_to_tensor,
                                                rank=rank)
        self.dt = 10
        self.set_seed(seed)
        self.config = parse_config(config_file)
        self.args = args
        self.model_id = model_id
        self.number_of_episodes = 0
        self.episode_over = False
        self.rank = rank
        self.ep_num=0
        self.agent_min_z = 0.05  # just for locobot
        self.agent_max_z = 0.9  # just for locobot
        self.camera_height = 0.857  # just for locobot
        self.mapper = self.build_mapper()

    def reset(self):
        self.timestep = 0
        self.ep_num += 1
        self.episode_train_over = np.array([False for _ in range(self.robots_num)])
        self.episode_test_over = np.array([False for _ in range(self.robots_num)])
        # get ground truth map
        self.explorable_map = None
        full_map_size = self.config.get('map_size_cm') // self.config.get('map_resolution')
        self.full_map_size = full_map_size
        while self.explorable_map is None:
            multi_state_depth_goal, multi_infos = super(CollaVN, self).reset()
            # multi_stat_depth is a list: index 0 contains a state and depth, index 1 contains goal panorama
            self.explorable_map = self._get_gt_map(full_map_size)

        # Preprocess observations
        multi_states = multi_state_depth_goal[0]['panorama_egomap'][0] # agent_num, 128, 512, 3  rgb
        multi_depths = multi_state_depth_goal[0]['panorama_egomap'][1]  # agent_num, 4, 128, 128, 1, unit m  # front, left, back, right  device: cuda
        goal_panorama = multi_state_depth_goal[1]  # 128, 512, 3
        # Initialize map and pose
        self.map_size_cm = self.config.get('map_size_cm')  # now 3840
        self.mapper.reset_map(self.map_size_cm)
        self.multi_curr_loc = self.get_sim_location()  # x, y ,o  # unit m  (c, r, o)
        # x,y (0, 0) means center of map, o [-pi, pi)
        self.multi_last_loc = self.multi_curr_loc
        multi_mapper_gt_pose = np.copy(self.multi_curr_loc)  # x, y ,o  (0, 0) is the center and o is rad
        self.explorable_map *= 100
        self.explorable_map = torch.flip(self.explorable_map, [1])
        multi_fp_projs, _, multi_fp_exploreds, _ = \
                self.mapper.update_map(multi_depths, multi_mapper_gt_pose)  # TODO: need to imple mapper.update_map
        # Initiallize variables
        self.scene_name = self.model_id
        self.collison_map = np.zeros(full_map_size)

        self.multi_infos = {
            'time': self.timestep,
            'goal_panorama': goal_panorama,
            'multi_fp_proj': multi_fp_projs,
            'multi_fp_explored': multi_fp_exploreds,
            'multi_sensor_pose': [[0., 0., 0.] for _ in range(self.robots_num)],
            'curr_pose': multi_mapper_gt_pose,
            'success': self.success,
            'rank': self.rank,
            'device': self.device_id,
            'reset': 1,
            'mask': np.array([1 for _ in range(self.robots_num)]),
            'final_distance': np.array([0. for _ in range(self.robots_num)]),
            'final_step': np.array([0 for _ in range(self.robots_num)]),
            'actual_length': np.array([0. for _ in range(self.robots_num)]),
            'ground_truth': np.array(self.potential),
        }
        # now rgbs and projs in cuda:id, we need to move to cuda
        done = torch.FloatTensor([0]).cuda(self.device_id)  # tensor([0.], device='cuda:0')
        ma_rew = torch.FloatTensor([0 for a in range(self.robots_num)]).cuda(self.device_id) # tensor([0., 0.], device='cuda:0')
        return multi_states, ma_rew, done, self.multi_infos

    def add_noise_to_action(self, ratio):
        pass

    def step(self, action_list):
        # x, y, o
        self.timestep += 1
        mask = 1 - self.episode_train_over

        self.multi_last_loc = np.copy(self.multi_curr_loc)   # t-1  # x, y ,o  # unit m  (c, r, o)
        self._previous_actionl_list = action_list
        ma_obs, ma_rew, done, ma_info = super(CollaVN, self).step(action_list)
        self.multi_curr_loc = self.get_sim_location()  # t  # x, y ,o  # unit m  (c, r, o)

        # Preprocess observations t
        ma_panoramas = ma_obs['panorama_egomap'][0]  # a, h, w, 3
        ma_depths = ma_obs['panorama_egomap'][1]  # a, 4, h, w, 1  # unit m
        self.ma_obs = ma_panoramas  # next time observation
        ma_state = ma_panoramas
        ma_rew = torch.FloatTensor(ma_rew).cuda(self.device_id)

        # get base sensor and ground-truth pose and update self.multi_curr_gt_loc
        # x: [-,+] y: [-,+] o: [0, 2pi]
        dc_gt_list, dr_gt_list, do_gt_list = self.get_gt_pose_change()
        dc_base_list, dr_base_list, do_base_list = self.get_base_pose_change(
                                                        action_list, (dc_gt_list, dr_gt_list, do_gt_list))
        ma_mapper_gt_pose = np.copy(self.multi_curr_loc)  # x, y ,o  (0, 0) is the center and o is rad
        # turn left means o +, turn right means o -
        # yaw is the third position is euler orientation

        multi_fp_projs, _, multi_fp_exploreds, _ = \
            self.mapper.update_map(ma_depths, ma_mapper_gt_pose)  # TODO: need to dobule check

        done, success, _ = self.get_termination()



        current_success_id = np.where(success * (1 - self.episode_test_over))[0]
        if len(current_success_id) > 0:
            print("----Success ", self.model_id, " ", current_success_id,  " ----", ma_rew[current_success_id], self.timestep, self.rank)

        current_done_id = np.where(done * (1 - self.episode_train_over))[0]

        if len(current_success_id) > 0:
            self.multi_infos['final_distance'][current_success_id] = np.array(ma_info['new_potential'])[current_success_id]
            self.multi_infos['final_step'][current_success_id] = self.timestep
        if self.timestep == self.max_step:
            timeout_id = np.where((1 - success))[0]
            self.multi_infos['final_distance'][timeout_id] = np.array(ma_info['new_potential'])[timeout_id]
            self.multi_infos['final_step'][timeout_id] = self.timestep
        self.episode_train_over = done.copy()
        self.episode_test_over = success.copy()

        
        done_tensor = torch.FloatTensor(done).cuda(self.device_id)
        # Set info
        self.multi_infos['time'] = self.timestep
        self.multi_infos['multi_fp_proj'] = multi_fp_projs  # a, h, w
        self.multi_infos['multi_fp_explored'] = multi_fp_exploreds  # a, h ,w
        self.multi_infos['multi_sensor_pose'] = [[dr_base_list[i], -dc_base_list[i], do_base_list[i]] for i in range(self.robots_num)]  # r,c,o
        self.multi_infos['multi_pose_err'] = [[0., 0., 0.] for i in range(self.robots_num)]
        self.multi_infos['curr_pose'] =  ma_mapper_gt_pose  # m
        self.multi_infos['success'] = self.success
        self.multi_infos['rank'] = self.rank
        self.multi_infos['device'] = self.device_id
        self.multi_infos['reset'] = 0
        self.multi_infos['mask'] = mask
        self.multi_infos['potential'] = ma_info['potential']
        self.multi_infos['new_potential'] = ma_info['new_potential']

        self.multi_infos['actual_length'] += np.abs(np.array(ma_info['new_potential']) - np.array(ma_info['potential'])) * (1-self.success)

        r = int(ma_mapper_gt_pose[0][1] * 10+192)
        c = 192 - int(ma_mapper_gt_pose[0][0] * 10)
        r = max(r, 0)
        r = min(r, self.explorable_map.shape[0]-1)
        c = max(c, 0)
        c = min(c, self.explorable_map.shape[1]-1)
        self.explorable_map[r, c] = 255
        return ma_state, ma_rew, done_tensor, self.multi_infos

    def get_global_reward(self):
        # TODO: tobe continue, maybe accumultation rewards
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.config.get('image_width', 128)
        params['frame_height'] = self.config.get('image_height', 128)
        params['fov'] = self.config.get('vertical_fov', 90)
        params['resolution'] = self.config.get('map_respolution', 10)
        params['map_size_cm'] = self.config.get('map_size_cm', 3840)
        params['agent_min_z'] = self.agent_min_z * 100  # cm
        params['agent_max_z'] = self.agent_max_z * 100  # cm
        params['agent_height'] = self.camera_height * 100  # cm
        params['agent_view_angle'] = 0  # TODO: temp manual set to 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.config.get('vision_range', 64)
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        params['device_id'] = self.device_id
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper

    def get_sim_location(self):
        # convert x from [-,+] to all+
        multi_agents_xyo = []
        for agent in self.robots:
            agent_pos = agent.get_position()  # (x, y, z) or (c, r, z), unit=m, (0, 0) is the center of map
            x, y = agent_pos[:2]
            agent_ori = agent.get_orientation()  # xyzw
            euler_ori = p.getEulerFromQuaternion(agent_ori)  # roll, pitch, yaw
            o = euler_ori[2]  # [-pi, pi) o is yaw
            if o < -np.pi:
                o += 2*np.pi
            elif o >= np.pi:
                o -= 2*np.pi
            multi_agents_xyo.append([x, y ,o])
        return np.array(multi_agents_xyo)


    def get_gt_pose_change(self):
        # return gt pose change and update self.multi_last_loc
        dx_gt_list = []
        dy_gt_list = []
        do_gt_list = []

        gt_pose_changel_list = self.multi_curr_loc - self.multi_last_loc
        for i in range(self.robots_num):
            dx_gt_list.append(gt_pose_changel_list[i][0])
            dy_gt_list.append(gt_pose_changel_list[i][1])
            do_gt_list.append(gt_pose_changel_list[i][2])
        return np.array(dx_gt_list), np.array(dy_gt_list), np.array(do_gt_list)

    def get_base_pose_change(self, action_list, gt_pose_change):
        dx_gt_list, dy_gt_list, do_gt_list = gt_pose_change
        x_err, y_err, o_err = 0., 0., 0.
        for i in range(self.robots_num):
            dx_gt_list[i] += x_err
            dy_gt_list[i] += y_err
            do_gt_list[i] += np.deg2rad(o_err)


        return dx_gt_list, dy_gt_list, do_gt_list

    def _get_gt_map(self, full_map_size):

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        self.scene_name = self.model_id
        sim_map = self.scene.floor_map[self.floor_num].copy()  # 0, 255 # 255 is trav_space
        sim_map[sim_map > 0] = 1.
        sim_map = torch.tensor(sim_map)
        self.ori_map_shape = sim_map.shape
        self.map_trans_ratio = (full_map_size / self.ori_map_shape[0])
        ori_size = self.ori_map_shape[0]
        if full_map_size > ori_size:
            episode_map[(full_map_size - ori_size) // 2:
                        (full_map_size - ori_size) // 2 + ori_size,
                        (full_map_size - ori_size) // 2:
                        (full_map_size - ori_size) // 2 + ori_size] = sim_map
        else:
            episode_map = sim_map[(ori_size - full_map_size)//2:
                              (ori_size - full_map_size)//2 + full_map_size,
                              (ori_size - full_map_size)//2:
                              (ori_size - full_map_size)//2 + full_map_size]
        return episode_map



