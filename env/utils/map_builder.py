import numpy as np
import torch

import env.utils.depth_utils as du


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        self.device_id = params['device_id']
        fov = params['fov']
        # depth frame width, height
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)  # return focal point coordinate, is also camera matrix
        self.vision_range = params['vision_range']
        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution']
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.z_bins = torch.tensor(self.z_bins).cuda(self.device_id).contiguous()
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']

        #  exp 384, 384, 3
        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)
        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

    def update_map(self, multi_depth, multi_current_pose):
        # mutli_depth agent_num, 4, 128 ,128 ,1  unit m
        # multi_current_pose agent, 3  x(m), y(m), o(rad)[-pi, pi)

        # convert depth unit from m to cm
        multi_depth *= 100.0
        agent_num = len(multi_current_pose)
        multi_agent_view_cropped = []
        multi_agent_view_explored = []
        for robot_id in range(agent_num):
            robot_current_pose = multi_current_pose[robot_id]  # x, y, o
            multidirec_agent_view_cropped = []
            multidirec_agent_view_explored = []
            for i in range(4):  # front left back right
                depth = multi_depth[robot_id][i, :, :, 0]  # depth unit cm
                # turn left is +, and turn right is -
                # current_pose = list(robot_current_pose).copy()
                # current_pose[0] *= 100.0
                # current_pose[1] *= 100.0
                # current_pose[2] = np.rad2deg((current_pose[2] - (i * np.pi / 2.0)) % (2 * np.pi))  TODO: no use now temp
                # depth unit cm
                with np.errstate(invalid="ignore"):
                    depth[depth > self.vision_range * self.resolution] = np.NaN
                point_cloud = du.get_point_cloud_from_z_tensor(depth, self.camera_matrix, \
                                                      scale=self.du_scale, device_id=self.device_id)
                # point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                #                         scale=self.du_scale)
                # print('point_cloud', point_cloud.shape) 64, 64, 3

                agent_view = du.transform_camera_view_tensor(point_cloud,
                                                      self.agent_height,
                                                      self.agent_view_angle, device_id=self.device_id)
                # agent_view = du.transform_camera_view(point_cloud,
                #                                              self.agent_height,
                #                                              self.agent_view_angle)
                # print('agent_view', agent_view.shape)  64, 64 ,3

                # shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]  # unit cm
                shift_loc = [self.vision_range * self.resolution // 2, self.vision_range * self.resolution, -np.pi / 2.0]  # unit cm
                agent_view_centered = du.transform_pose_tensor(agent_view, shift_loc, device_id=self.device_id)
                # agent_view_centered = du.transform_pose(agent_view, shift_loc)
                # print('agent_view_centered', agent_view_centered.shape)  64, 64 ,3
                agent_view_flat = du.bin_points_tensor(
                    agent_view_centered,
                    self.vision_range,
                    self.z_bins,
                    self.resolution)
                # agent_view_flat = du.bin_points(
                #     agent_view_centered,
                #     self.vision_range,
                #     self.z_bins,
                #     self.resolution)
                # print('agent_view_flat', agent_view_flat.shape) 64, 64 ,3

                agent_view_cropped = agent_view_flat[:, :, 1]

                agent_view_cropped = agent_view_cropped / self.obs_threshold
                agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
                agent_view_cropped[agent_view_cropped < 0.5] = 0.0

                agent_view_explored = agent_view_flat.sum(2)
                agent_view_explored[agent_view_explored > 0] = 1.0

                multidirec_agent_view_cropped.append(agent_view_cropped)
                multidirec_agent_view_explored.append(agent_view_explored)
                # import cv2 as cv
                # if robot_id == 0:
                #     cv.imwrite('/home/why/Desktop/crop'+str(robot_id) +'_' + str(i)+'.png', np.array((agent_view_cropped.cpu()*255)).astype(np.uint8))
                #     cv.imwrite('/home/why/Desktop/explored'+str(robot_id)+ '_' + str(i) + '.png',
                #             np.array((agent_view_explored.cpu()*255)).astype(np.uint8))
                #     cv.imwrite('/home/why/Desktop/depth'+str(robot_id)+'_' + str(i)+'.png', np.array(depth.cpu()/500.0*255).astype(np.uint8))
                # geocentric_pc may no use, so not centered
                # geocentric_pc = du.transform_pose_tensor(agent_view, current_pose)

                # geocentric_flat = du.bin_points_tensor(
                #     geocentric_pc,
                #     self.map.shape[0],
                #     self.z_bins,
                #     self.resolution)
                # self.map = self.map + geocentric_flat
            merge_agent_view_cropped, merge_agent_view_explored =\
                               du.merge_geocentric_map_tensor(multidirec_agent_view_cropped,
                                                       multidirec_agent_view_explored, device_id=self.device_id)
            # if robot_id == 0:
            #     import cv2 as cv
            #     cv.imwrite('/home/why/Desktop/mergecrop' + str(robot_id) + '.png',
            #            np.array(merge_agent_view_cropped.cpu()*255).astype(np.uint8))
            #     cv.imwrite('/home/why/Desktop/mergeexplore' + str(robot_id) + '.png',
            #                np.array(merge_agent_view_explored.cpu() * 255).astype(np.uint8))
            multi_agent_view_cropped.append(merge_agent_view_cropped)
            multi_agent_view_explored.append(merge_agent_view_explored)
        # map_gt maybe no use
        # map_gt = self.map[:, :, 1] / self.obs_threshold
        # map_gt[map_gt >= 0.5] = 1.0
        # map_gt[map_gt < 0.5] = 0.0

        # explored_gt = self.map.sum(2)
        # explored_gt[explored_gt > 1] = 1.0

        multi_agent_view_cropped = torch.stack(multi_agent_view_cropped, dim=0)
        multi_agent_view_explored = torch.stack(multi_agent_view_explored, dim=0)

        return multi_agent_view_cropped, None, multi_agent_view_explored, None

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
