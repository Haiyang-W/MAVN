import gibson2
from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW, cartesian_to_polar
from env.base_env import Ma_BaseEnv
from env.utils.utils import batch_l2_distance
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from transforms3d.quaternions import quat2mat, qmult
import gym
import numpy as np
import os
import pybullet as p
from IPython import embed
import cv2
import time
import collections
import time
import logging
import random
import json

class MaBase_ImgGoal_NavigateEnv(Ma_BaseEnv):
    """
    We define a Multi agent Image Goal Navigation task environment.
    Fixed initialization and targets.
    """
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
            render_to_tensor=False,
            rank=0
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(MaBase_ImgGoal_NavigateEnv, self).__init__(config_file=config_file,
                                          model_id=model_id,
                                          mode=mode,
                                          action_timestep=action_timestep,
                                          physics_timestep=physics_timestep,
                                          device_idx=device_idx,
                                          render_to_tensor=render_to_tensor)
        self.automatic_reset = automatic_reset
        self.device_id = device_idx
        self.rank = rank

    def load_task_setup(self):
        """
        Load task setup, including initialization, termination condition, reward, collision checking, discount factor
        """
        # initial and target pose
        # the initial pos in config should contain multi agents setting
        self.task_type = self.config.get('task_type', "common")

        self._trav_map_resolution = self.config.get('trav_map_resolution', 0.05)
        self._trav_map_size_cm = self.config.get('map_size_cm', 3600)
        half_blen_m = self._trav_map_size_cm / 100. / 2.
        self.position_boundry = [-half_blen_m, half_blen_m]

        self.initial_pos_z_offset = self.config.get('initial_pos_z_offset', [0.1])  # is a list
        check_collision_distance = self.initial_pos_z_offset * 0.5
        # s = 0.5 * G * (t ** 2)
        check_collision_distance_time = np.sqrt(check_collision_distance / (0.5 * 9.8))
        self.check_collision_loop = (check_collision_distance_time / self.physics_timestep).astype(np.int)

        self.additional_states_dim = self.config.get('additional_states_dim', 0)
        self.goal_format = self.config.get('goal_format', 'polar')

        # termination condition
        self.dist_tol = self.config.get('dist_tol', 0.5)
        self.dist_tol_train = self.config.get('dist_tol_train', 0.36)
        self.max_step = self.config.get('max_episode_length', 80)
        self.max_collisions_allowed = self.config.get('max_collisions_allowed', 500)

        # reward
        self.reward_type = self.config.get('reward_type', 'l2')
        assert self.reward_type in ['geodesic', 'l2', 'sparse']

        self.success_reward = self.config.get('success_reward', 10.0)
        self.slack_reward = self.config.get('slack_reward', -0.01)

        # reward weight
        self.potential_reward_weight = self.config.get('potential_reward_weight', 1.0)
        self.collision_reward_weight = self.config.get('collision_reward_weight', -0.1)

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        self.reset_file_root = self.config.get('reset_file_root')

        self.reset_file = os.path.join(self.reset_file_root, str(self.robots_num), self.model_id+'.json')

        with open(self.reset_file, 'r') as f:
            self.reset_list = json.load(f)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        self.panorama_width = self.config.get('panorama_width', 512)
        self.panorama_height = self.config.get('panorama_height', 128)
        self.vision_range = self.config.get('vision_range', 64)
        observation_space = OrderedDict()
        if 'sensor' in self.output:
            self.sensor_dim = self.additional_states_dim
            self.sensor_space = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(self.sensor_dim,),
                                               dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'panorama_egomap' in self.output:
            self.panorama_space = gym.spaces.Box(low=0.0,
                                                 high=1.0,
                                                 shape=(self.panorama_height, self.panorama_width, 3),
                                                 dtype=np.float32)
            self.ego_space = gym.spaces.Box(low=0.0,
                                                 high=1.0,
                                                 shape=(self.vision_range, self.vision_range, 1),
                                                 dtype=np.float32)
            self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)
            self.depth_low = self.config.get('depth_low', 0.5)
            self.depth_high = self.config.get('depth_high', 5.0)
            observation_space['panorama'] = self.panorama_space
            observation_space['egomap'] = self.ego_space

        if 'depth' in self.output:
            self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)
            self.depth_low = self.config.get('depth_low', 0.5)
            self.depth_high = self.config.get('depth_high', 5.0)
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.image_height, self.image_width, 1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'rgbd' in self.output:
            self.rgbd_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.image_height, self.image_width, 4),
                                             dtype=np.float32)
            observation_space['rgbd'] = self.rgbd_space
        if 'seg' in self.output:
            self.seg_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.image_height, self.image_width, 1),
                                            dtype=np.float32)
            observation_space['seg'] = self.seg_space
        if 'scan' in self.output:
            self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            self.laser_linear_range = self.config.get('laser_linear_range', 10.0)
            self.laser_angular_range = self.config.get('laser_angular_range', 180.0)
            self.min_laser_dist = self.config.get('min_laser_dist', 0.05)
            self.laser_link_name = self.config.get('laser_link_name', 'scan_link')
            self.scan_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
        if 'rgb_filled' in self.output:  # use filler
            try:
                import torch.nn as nn
                import torch
                from torchvision import datasets, transforms
                from gibson2.learn.completion import CompletionNet
            except:
                raise Exception('Trying to use rgb_filled ("the goggle"), but torch is not installed. Try "pip install torch torchvision".')

            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda(self.device_id)
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        self.observation_space = gym.spaces.Dict(observation_space)

    def load_action_space(self):
        """
        Load action space
        Homogeneous agents, so the action space is identical
        Continious action in CollaVN
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = np.zeros(self.robots_num).astype(np.int)
        self.collision_step = np.zeros(self.robots_num).astype(np.int)
        self.current_episode = 0
        self.floor_num = 0

    def load(self):
        """
        Load multi agents navigation environment
        """
        super(MaBase_ImgGoal_NavigateEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def global_to_local(self, pos):
        """
        Convert a 3D point in global frame to agent's local frame
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return [rotate_vector_3d(pos[i] - self.robots[i].get_position(), *self.robots[i].get_rpy()) for i in range(self.robots_num)]


    def add_naive_noise_to_sensor(self, sensor_reading, noise_rate, noise_value=1.0):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan
        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if noise_rate <= 0.0:
            return sensor_reading

        assert len(sensor_reading[(sensor_reading < 0.0) | (sensor_reading > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'

        valid_mask = np.random.choice(2, sensor_reading.shape, p=[noise_rate, 1.0 - noise_rate])
        sensor_reading[valid_mask == 0] = noise_value
        return sensor_reading

    def get_depth(self):
        """
        :return: depth sensor reading, normalized to [0.0, 1.0]
        """
        depths = self.simulator.renderer.render_robot_cameras(modes=('3d'))
        new_depths = []
        for depth in depths:
            # 0.0 is a special value for invalid entries
            depth = -depth[:, :, 2:3]
            depth[depth < self.depth_low] = 0.0
            depth[depth > self.depth_high] = 0.0

            # re-scale depth to [0.0, 1.0]
            depth /= self.depth_high
            depth = self.add_naive_noise_to_sensor(depth, self.depth_noise_rate, noise_value=0.0)
            new_depths.append(depth)
        return new_depths

    def get_rgb(self):
        """
        :return: RGB sensor reading, normalized to [0.0, 1.0], a frame list for multi agents
        """
        rgbs = self.simulator.renderer.render_robot_cameras(modes=('rgb'))
        return [rgb[:, :, :3] for rgb in rgbs]

    def get_panorama_multidepth(self):
        """
        :return: four direction
        """
        panoramas, multi_depths = self.simulator.renderer.render_robot_panorama_depth(self.panorama_height, self.panorama_width, self.image_width, True, self.device_id)
        # a ,512, 128, 3
        # a, 4, 128 ,128, 1
        multi_depths = -multi_depths
        multi_depths[multi_depths < self.depth_low] = 0.0
        multi_depths[multi_depths > self.depth_high] = 0.0
        return [panoramas, multi_depths]

    def get_pc(self):
        """
        :return: pointcloud sensor reading, a frame list for multi agents
        """
        # return self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
        return self.simulator.renderer.render_robot_cameras(modes=('3d'))

    def get_normal(self):
        """
        :return: surface normal reading, a frame list for multi agents
        """
        # return self.simulator.renderer.render_robot_cameras(modes='normal')
        return self.simulator.renderer.render_robot_cameras(modes='normal')

    def get_seg(self):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0], a frame list for multi agents
        """
        # seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')
        seg = [seg_i[:, :, 0:1] for seg_i in seg]
        if self.num_object_classes is not None:
            # seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
            seg = [np.clip(seg_i * 255.0 / self.num_object_classes, 0.0, 1.0) for seg_i in seg]
        return seg

    def get_scan(self):
        """
        :return: LiDAR sensor reading, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        laser_pose = self.robots[0].parts[self.laser_link_name].get_pose()
        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                          laser_angular_half_range / 180 * np.pi,
                          self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_local = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        scans = []
        for i in range(self.robots_num):
            if self.laser_link_name not in self.robots[i].parts:
                raise Exception('Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples.')
            transform_matrix = quat2mat([laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
            unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

            start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
            start_pose += unit_vector_world * self.min_laser_dist
            end_pose = laser_pose[:3] + unit_vector_world * self.laser_linear_range
            results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6

            hit_fraction = np.array([item[2] for item in results])  # hit fraction = [0.0, 1.0] of self.laser_linear_range
            hit_fraction = self.add_naive_noise_to_sensor(hit_fraction, self.scan_noise_rate)
            scan = np.expand_dims(hit_fraction, 1)
            scans.append(scan)
        return scans

    def get_state(self, collision_links=[]):
        """
        :param collision_links: collisions from last time step
        :return: observation as a dictionary for multi agent
        """
        state = OrderedDict()
        if 'sensor' in self.output:
            state['sensor'] = self.get_additional_states()
        if 'rgb' in self.output:
            state['rgb'] = self.get_rgb()
        if 'panorama_egomap' in self.output:
            state['panorama_egomap'] = self.get_panorama_multidepth()
        if 'depth' in self.output:
            state['depth'] = self.get_depth()
        if 'pc' in self.output:
            state['pc'] = self.get_pc()
        if 'rgbd' in self.output:
            rgb = self.get_rgb()
            depth = self.get_depth()
            state['rgbd'] = np.concatenate((rgb, depth), axis=2)
        if 'normal' in self.output:
            state['normal'] = self.get_normal()
        if 'seg' in self.output:
            state['seg'] = self.get_seg()
        if 'rgb_filled' in self.output:
            state['rgb_filled'] = []
            with torch.no_grad():
                for state_rgb_i in state['rgb']:
                    tensor = transforms.ToTensor()((state_rgb_i * 255).astype(np.uint8)).cuda(self.device_id)
                    rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                    state['rgb_filled'].append(rgb_filled)
        if 'scan' in self.output:
            state['scan'] = self.get_scan()
        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (simulator_loop physics timestep)
        :return: collisions from this simulation
        """

        multi_agent_collision_links = [[] for _ in range(self.robots_num)]
        for _ in range(self.simulator_loop):
            self.simulator_step()
            for i in range(self.robots_num):
                multi_agent_collision_links[i].append(list(p.getContactPoints(bodyA=self.robots[i].robot_ids[0])))  # TODO: robot_ids[0] not sure
        self.simulator.sync()

        return self.filter_collision_links(multi_agent_collision_links)

    def filter_collision_links(self, multi_agent_collision_links):
        """
        Filter out collisions that should be ignored
        :param mutli_agent_collision_links: original collisions, a list of lists of collisions for multi agent
        :return: filtered collisions
        """
        new_multi_agent_collision_links = []
        for i, collision_links in enumerate(multi_agent_collision_links):
            new_collision_links = []
            for collision_per_sim_step in collision_links:
                new_collision_per_sim_step = []
                for item in collision_per_sim_step:
                    # ignore collision with body b
                    if item[2] in self.collision_ignore_body_b_ids:
                        continue
                    # ignore collision with robot link a
                    if item[3] in self.collision_ignore_link_a_ids:
                        continue
                    # ignore self collision with robot link a (body b is also robot itself)
                    if item[2] == self.robots[i].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                        continue
                    new_collision_per_sim_step.append(item)
                new_collision_links.append(new_collision_per_sim_step)
            new_multi_agent_collision_links.append(new_collision_links)
        return new_multi_agent_collision_links

    def get_position_of_interest(self):
        """
        Get position of interest.
        :return: If pointgoal task, return multi agent base position. If reaching task, return multi agent end effector position.
        """

        return [robot.get_position() for robot in self.robots]


    def get_shortest_path(self, from_initial_pos=False, entire_path=False):
        """
        :param from_initial_pos: whether source is initial positions rather than current positions for multi agents
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position - multi agent,  is a list []
        """
        if from_initial_pos:
            sources = [init_pos[:2] for init_pos in self.initial_pos]
        else:
            sources = [robot.get_position()[:2] for robot in self.robots]
        if self.task_type == 'common':
            short_paths = [self.scene.get_shortest_path(self.floor_num, sources[i], self.target_pos[0][:2], entire_path=entire_path) for i in range(self.robots_num)]
        elif self.task_type == 'specific':
            short_paths = [self.scene.get_shortest_path(self.floor_num, sources[i], self.target_pos[i][:2], entire_path=entire_path) for i in range(self.robots_num)]
        return short_paths

    def get_shortest_path_id(self, from_initial_pos=False, entire_path=False, agent_ids=[]):
        """
        :param from_initial_pos: whether source is initial positions rather than current positions for multi agents
        :param entire_path: whether to return the entire shortest path
        :param agent_id_list: ndarray
        :return: shortest path and geodesic distance to the target position - multi agent,  is a list []
        """
        if len(agent_ids) == 0:
            agent_ids = np.arange(self.robots_num)
        if from_initial_pos:
            sources = np.array(self.initial_pos)[agent_ids][:, :2]
        else:
            sources = []
            for i, robot in enumerate(self.robots):
                if (i in agent_ids):
                    sources.append(robot.get_position()[:2])
            sources = np.array(sources)
        if self.task_type == 'common':
            short_paths = [self.scene.get_shortest_path(self.floor_num, sources[i],
                                                        self.target_pos[0][:2], entire_path=entire_path) for i in range(len(sources))]
        elif self.task_type == 'specific':
            short_paths = [self.scene.get_shortest_path(self.floor_num, sources[i],
                                                        self.target_pos[i][:2], entire_path=entire_path) for i in range(len(sources))]

        return short_paths

    def get_geodesic_potential(self):
        """
        :return: geodesic distance to the target position
        """
        short_paths = self.get_shortest_path()
        geodesic_dist = [short_path[1] for short_path in short_paths]  # for multi agent, it's a list
        return geodesic_dist

    def get_l2_potential(self):
        """
        :return: L2 distance to the target position for multi agent setting
        """
        get_position_of_interest = self.get_position_of_interest()
        if self.task_type == 'common':
            return [l2_distance(self.target_pos[0], get_position_of_interest[i]) for i in range(self.robots_num)]
        elif self.task_type == 'specific':
            return [l2_distance(self.target_pos[i], get_position_of_interest[i]) for i in range(self.robots_num)]

    def is_goal_reached(self):
        """
        Real goal reached detection in real simulation or evaluation, dist_tol: 1.
        :return: L2 distance to the target position for multi agent setting
        """
        reached_results = []
        agent_ids = []
        # first l2 distance
        multi_l2_distance = self.get_l2_potential()
        for i in range(self.robots_num):
            if multi_l2_distance[i] < self.dist_tol:
                reached_results.append(True)
                agent_ids.append(i)
            else:
                reached_results.append(False)

        if len(agent_ids) == 0:
            return reached_results
        # Second geodesic distance if need
        agent_ids = np.array(agent_ids)

        part_short_paths = self.get_shortest_path_id(from_initial_pos=False, entire_path=False, agent_ids=agent_ids)
        dist = np.array([short_path[1] for short_path in part_short_paths])
        for i, gd in enumerate(dist):
            if gd >= self.dist_tol:
                reached_results[agent_ids[i]] = False
        return reached_results

    def is_goal_reached_train(self):
        """
        Real goal reached detection in training stage, dist_tol: 0.36
        :return: L2 distance to the target position for multi agent setting
        """
        reached_results = []
        agent_ids = []
        # first l2 distance
        multi_l2_distance = self.get_l2_potential()
        for i in range(self.robots_num):
            if multi_l2_distance[i] < self.dist_tol_train:
                reached_results.append(True)
                agent_ids.append(i)
            else:
                reached_results.append(False)

        if len(agent_ids) == 0:
            return reached_results
        # Second geodesic distance if need
        agent_ids = np.array(agent_ids)

        part_short_paths = self.get_shortest_path_id(from_initial_pos=False, entire_path=False, agent_ids=agent_ids)
        dist = np.array([short_path[1] for short_path in part_short_paths])
        for i, gd in enumerate(dist):
            if gd >= self.dist_tol_train:
                reached_results[agent_ids[i]] = False
        return reached_results

    def get_reward(self, multi_agent_collision_links=[], action=None, info={}):
        """
        :param collision_links: collisions from last time step for multi agent setting
        :param action: last action
        :param info: a dictionary to store additional info
        :return: reward, info

        reward: slack reward, collision_reward, find goal reward, distance reward
        """
        multi_agent_collision_links_flatten = []
        for collision_links in multi_agent_collision_links:
            multi_agent_collision_links_flatten.append([item for sublist in collision_links for item in sublist])
        rewards = [self.slack_reward for _ in range(self.robots_num)]  # |slack_reward| = 0.01 per step
        if self.reward_type == 'l2':
            new_potential = self.get_l2_potential()
        elif self.reward_type == 'geodesic':
            new_potential = self.get_geodesic_potential()
        else:
            raise Exception(
                'reward_type is wrong, only support l2 or geodesic')
        # potential_reward = self.potential - new_potential
        potential_rewards = [self.potential[i] - new_potential[i] for i in range(self.robots_num)]
        info['new_potential'] = new_potential
        info['potential'] = self.potential

        potential_rewards = [p_r_i * self.potential_reward_weight for p_r_i in potential_rewards]
        rewards = [rewards[i] + potential_rewards[i] for i in
                   range(self.robots_num)]  # |potential_reward| ~= 0.1 per step
        self.potential = new_potential
        multi_agent_collision_reward = [float(len(collision_links_flatten) > 0) for collision_links_flatten in
                                        multi_agent_collision_links_flatten]
        self.collision_step += np.array(multi_agent_collision_reward).astype(np.int)
        multi_agent_collision_reward = [collision_reward * self.collision_reward_weight for collision_reward in
                                        multi_agent_collision_reward]  # |collision_reward| ~= 1.0 per step if collision
        rewards = [rewards[i] + multi_agent_collision_reward[i] for i in range(self.robots_num)]

        reach_results = self.is_goal_reached_train()
        
        for i in range(self.robots_num):
            if reach_results[i] and not self.finish_for_reward:
                rewards[i] += self.success_reward
                self.finish_for_reward = 1
        return rewards, info

    def get_termination(self, collision_links=[], action=None, info={}):
        """
        :param collision_links: collisions from last time step
        :param info: a dictionary to store additional info
        :return: done, info
        """

        # goal reached
        reach_results = np.array(self.is_goal_reached())
        temp_success = reach_results.astype(np.uint8)
        collision_mask = np.array(self.collision_step) < self.max_collisions_allowed
        temp_success *= collision_mask
        self.success[np.where(temp_success)[0]] = 1

        train_reach_results = np.array(self.is_goal_reached_train())
        temp_done = train_reach_results.astype(np.uint8)
        temp_done *= collision_mask
        self.done[np.where(temp_done)[0]] = 1

        done = self.done.copy()

        maxstep_mask =  self.current_step >= self.max_step
        done[np.where(maxstep_mask)[0]] = 1

        info['success'] = self.success
        info['done'] = done
        return done, self.success, info

    def before_simulation(self):
        """
        Cache bookkeeping data before simulation
        :return: cache
        """
        return {'robot_position': [self.robots[i].get_position() for i in range(self.robots_num)]}

    def after_simulation(self, cache, collision_links):
        """
        Accumulate evaluation stats
        :param cache: cache returned from before_simulation
        :param collision_links: collisions from last time step
        """
        old_robot_position = [pos[:2] for pos in cache['robot_position']]  # for multi agent, so it is a list
        new_robot_position = [robot.get_position()[:2] for robot in self.robots]
        self.path_length += batch_l2_distance(old_robot_position, new_robot_position)

    def step(self, action_list):
        """
        apply robot's action and get state, reward, done and info, following OpenAI gym's convention
        only for continous action
        :param action: a list of control signals
        :return: state, reward, done, info
        """

        self.current_step += 1
        for i in range(self.robots_num):
            if action_list[i] is not None:
                self.robots[i].apply_action(action_list[i])
        # caches = self.before_simulation()
        collision_links = self.run_simulation()
        # self.after_simulation(caches, collision_links)

        states = self.get_state(collision_links)
        info = {}
        rewards, info = self.get_reward(collision_links, action_list, info)
        done, success, info = self.get_termination(collision_links, action_list, info)

        return states, rewards, done, info

    def generate_target_image(self):
        goal_panoramas = self.simulator.renderer.render_panorama_with_position_oritentation(self.target_pos, self.target_orn, self.panorama_height,
                                                                                      self.panorama_width,
                                                                                       self.image_width, True, self.device_id)
        return goal_panoramas

    def reset_initial_and_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos
        """
        return

    def reset_initial_pos(self, robot_id):
        """
        Reset initial_pos, initial_orn and target_pos
        """
        return

    def reset_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos
        """
        return

    def check_collision(self, body_id):
        """
        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        for _ in range(self.check_collision_loop):
            self.simulator_step()
            collisions = list(p.getContactPoints(bodyA=body_id))

            if logging.root.level <= logging.DEBUG: #Only going into this if it is for logging --> efficiency
                for item in collisions:
                    logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
            if len(collisions) > 0:
                return False
        return True

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        obj.set_position_orientation([pos[0], pos[1], pos[2] + offset],
                                     quatToXYZW(euler2quat(*orn), 'wxyz'))

    def test_valid_position(self, obj_type, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id
        return self.check_collision(body_id)

    def land(self, obj_type, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.physics_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if obj_type == 'robot':
            obj.robot_specific_reset()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = np.zeros(self.robots_num).astype(np.int)
        self.collision_step = np.zeros(self.robots_num).astype(np.int)
        self.path_length = np.zeros(self.robots_num)
        self.geodesic_dist = self.get_geodesic_potential()

    def reset_agent(self):
        """
        Reset the robot's joint configuration and base pose until no collision
        """
        # print("---------------Reset Agent in Scene Rank ", self.rank, "------------------")
        self.initial_pos = np.array(self.reset_list[str(self.current_episode)]['init_pos'])
        self.initial_orn = np.array([[0.,0.,0.] for _ in range(self.robots_num)])
        self.target_pos = np.array(self.reset_list[str(self.current_episode)]['target_pos'])
        self.target_orn = np.array(self.reset_list[str(self.current_episode)]['target_orn'])

        for robot_id in range(self.robots_num):
            self.land('robot', self.robots[robot_id], self.initial_pos[robot_id], self.initial_orn[robot_id])
        return True

    def reset(self):
        """
        Reset episode
        """
        self.finish_for_reward = 0
        self.floor_num = self.reset_list[str(self.current_episode)]['floor_num']
        self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)

        success = self.reset_agent()
        goal_panorama = self.generate_target_image()
        self.success = np.zeros(self.robots_num).astype(np.uint8)
        self.done = np.zeros(self.robots_num).astype(np.uint8)
        self.simulator.sync()
        state = self.get_state()
        if self.reward_type == 'l2':
            self.potential = self.get_l2_potential()
        elif self.reward_type == 'geodesic':
            self.potential = self.get_geodesic_potential()
        self.reset_variables()

        state_goal = (state, goal_panorama)
        infos = {}

        return state_goal, infos


