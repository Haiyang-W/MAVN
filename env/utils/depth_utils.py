# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace

import numpy as np
import torch

import env.utils.rotation_utils as ru


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.  # x axis center
    zc = (height - 1.) / 2.  # y axis center
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))  # focal length
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    # x, z = np.meshgrid(np.arange(Y.shape[-1]),
    #                    np.arange(Y.shape[-2] - 1, -1, -1))
    x, z = np.meshgrid(np.arange(Y.shape[-1]-1, -1, -1),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ

def get_point_cloud_from_z_tensor(Y, camera_matrix, scale=1, device_id=0):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    # x, z = torch.meshgrid(torch.arange(Y.shape[-1]),
    #                    torch.arange(Y.shape[-2] - 1, -1, -1))
    x, z = torch.meshgrid(torch.arange(Y.shape[-1] - 1, -1, -1),
                          torch.arange(Y.shape[-2] - 1, -1, -1))
    x = torch.transpose(x, 0, 1).float().cuda(device_id)
    z = torch.transpose(z, 0, 1).float().cuda(device_id)
    for i in range(Y.dim() - 2):
        x = x.unsqueeze(0)
        z = z.unsqueeze(0)

    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
    XYZ = torch.cat((X.unsqueeze(-1), Y[::scale, ::scale].unsqueeze(-1),
                          Z.unsqueeze(-1)), dim=X.dim())
    return XYZ

def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ

def transform_camera_view_tensor(XYZ, sensor_height, camera_elevation_degree, device_id):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    R = torch.FloatTensor(R).cuda(device_id)
    XYZ = torch.matmul(XYZ.view(-1, 3), R.t()).view(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ

def transform_pose_tensor(XYZ, current_pose, device_id):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    R = torch.FloatTensor(R).cuda(device_id)
    XYZ = torch.matmul(XYZ.view(-1, 3), R.t()).view(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    isvalids = []
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)
        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        counts = np.reshape(count, [map_size, map_size, n_z_bins])

    counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts

def bin_points_tensor(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.view([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    isvalids = []
    for XYZ_cm in XYZ_cms:
        isnotnan = torch.logical_not(torch.isnan(XYZ_cm[:, :, 0]))
        X_bin = torch.round(XYZ_cm[:, :, 0] / xy_resolution).int()
        Y_bin = torch.round(XYZ_cm[:, :, 1] / xy_resolution).int()
        Z_bin = torch.bucketize(XYZ_cm[:, :, 2].contiguous(), z_bins, right=True).int()

        isvalid = torch.stack([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan], dim=0).bool()
        isvalid = isvalid.all(dim=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[torch.logical_not(isvalid)] = 0
        count = torch.bincount(ind.flatten(), isvalid.flatten().int(),
                            minlength=map_size * map_size * n_z_bins)
        counts = count.view([map_size, map_size, n_z_bins])

    counts = counts.view(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts

def merge_geocentric_map(multi_agent_crop, multi_agent_explored):
    # front, left, back, right
    range_size = multi_agent_crop[0].shape[0]
    merge_agent_view_cropped = np.zeros(multi_agent_crop[0].shape)
    merge_agent_view_explored = np.zeros(multi_agent_explored[0].shape)
    # sample range
    r_s, r_e, c_s, c_e = range_size//2, range_size, 0, range_size
    # front. left, back, right
    trans_range = [[0, range_size//2, 0, range_size], [0, range_size, 0, range_size//2],
                   [range_size//2, range_size, 0, range_size], [0, range_size, range_size//2, range_size]]

    for i in range(len(multi_agent_crop)):
        trans_r_s, trans_r_e, trans_c_s, trans_c_e = trans_range[i]
        rot_multi_agent_crop = multi_agent_crop[i][r_s:r_e, c_s:c_e]
        rot_multi_agent_explored = multi_agent_explored[i][r_s:r_e, c_s:c_e]
        for _ in range(i):
            rot_multi_agent_crop = np.rot90(rot_multi_agent_crop)
            rot_multi_agent_explored = np.rot90(rot_multi_agent_explored)
        merge_agent_view_cropped[trans_r_s: trans_r_e,
                                trans_c_s: trans_c_e] += rot_multi_agent_crop
        merge_agent_view_explored[trans_r_s: trans_r_e,
                                trans_c_s: trans_c_e] += rot_multi_agent_explored

    merge_agent_view_cropped[merge_agent_view_cropped >= 0.5] = 1.0
    merge_agent_view_cropped[merge_agent_view_cropped < 0.5] = 0.0
    merge_agent_view_explored[merge_agent_view_explored > 0.] = 1.0

    return merge_agent_view_cropped, merge_agent_view_explored

def merge_geocentric_map_tensor(multi_agent_crop, multi_agent_explored, device_id):
    # front, left, back, right
    range_size = multi_agent_crop[0].shape[0]
    merge_agent_view_cropped = torch.zeros_like(multi_agent_crop[0]).cuda(device_id)
    merge_agent_view_explored = torch.zeros_like(multi_agent_explored[0]).cuda(device_id)
    # sample range
    r_s, r_e, c_s, c_e = range_size//2, range_size, 0, range_size
    # front. left, back, right
    trans_range = [[0, range_size//2, 0, range_size], [0, range_size, 0, range_size//2],
                   [range_size//2, range_size, 0, range_size], [0, range_size, range_size//2, range_size]]

    for i in range(len(multi_agent_crop)):
        trans_r_s, trans_r_e, trans_c_s, trans_c_e = trans_range[i]
        rot_multi_agent_crop = multi_agent_crop[i][r_s:r_e, c_s:c_e]
        rot_multi_agent_explored = multi_agent_explored[i][r_s:r_e, c_s:c_e]
        for _ in range(i):
            rot_multi_agent_crop = torch.rot90(rot_multi_agent_crop)
            rot_multi_agent_explored = torch.rot90(rot_multi_agent_explored)
        merge_agent_view_cropped[trans_r_s: trans_r_e,
        trans_c_s: trans_c_e] += rot_multi_agent_crop
        merge_agent_view_explored[trans_r_s: trans_r_e,
        trans_c_s: trans_c_e] += rot_multi_agent_explored

    merge_agent_view_cropped[merge_agent_view_cropped >= 0.5] = 1.0
    merge_agent_view_cropped[merge_agent_view_cropped < 0.5] = 0.0
    merge_agent_view_explored[merge_agent_view_explored > 0.] = 1.0

    return merge_agent_view_cropped, merge_agent_view_explored





