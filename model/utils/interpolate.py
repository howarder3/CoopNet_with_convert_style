from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def interpolator(z, interp_type='linear'):
    if interp_type == 'linear':
        interp_z = linear_interpolator(z)
    elif interp_type == 'sphere':
        interp_z = sphere_interpolator(z)
    elif interp_type == 'both':
        interp_z = linear_interpolator(z)
    else:
        raise NotImplementedError

    return interp_z


def linear_interpolator(z, npairs=8, ninterp=8):
    
    num_z = z.shape[0]
    z_dim = z.shape[1]





    interp_z = np.zeros(shape=(npairs * ninterp, z_dim))
    # line_points = np.expand_dims(np.linspace(0, 1, ninterp), 0)
    line_points = np.expand_dims(np.linspace(0, 3 ,64, 64), 0)
    for ip in range(npairs):
        pair = np.random.permutation(num_z)
        l_z = z[np.newaxis, pair[0]]
        r_z = z[np.newaxis, pair[1]]

        print('z.shape = {}'.format(z.shape)) # z.shape = (100, 64, 64, 3)
        print('num_z = {}'.format(num_z)) # num_z = 100
        print('z_dim = {}'.format(z_dim)) # z_dim = 64
        print('npairs = {}'.format(npairs)) # npairs = 12
        print('ninterp = {}'.format(ninterp)) # ninterp = 12
        print('interp_z.shape = {}'.format(interp_z.shape)) # interp_z.shape = (144, 64)
        print('line_points.shape = {}'.format(line_points .shape)) # line_points.shape = (1, 12)
        print('line_points.transpose().shape = {}'.format(line_points.transpose().shape)) # line_points.transpose().shape = (12, 1)
        print('l_z.shape = {}'.format(l_z.shape)) # l_z.shape = (1, 64, 64, 3)
        print('1 - line_points.transpose().shape = {}'.format((1 - line_points.transpose()).shape)) # 1 - line_points.transpose().shape = (12, 1)
        print('r_z.shape = {}'.format(r_z.shape)) # r_z.shape = (1, 64, 64, 3)



        temp = np.dot(line_points.transpose(), l_z) + np.dot(1 - line_points.transpose(), r_z)
        interp_z[ip * ninterp:(ip + 1) * ninterp] = temp
    return interp_z


def sphere_interpolator(z, n_phi=8, n_theta=8):
    num_z = z.shape[0]
    z_dim = z.shape[1]
    interp_z = np.zeros(shape=(n_phi * n_theta, z_dim))
    phi_points = np.expand_dims(np.linspace(0, math.pi / 2, n_phi), 0)
    theta_points = np.linspace(0, math.pi / 2, n_theta)
    corner_idx = np.random.permutation(num_z)
    lu_z = z[np.newaxis, corner_idx[0]]
    ru_z = z[np.newaxis, corner_idx[1]]
    ld_z = z[np.newaxis, corner_idx[2]]
    rd_z = z[np.newaxis, corner_idx[3]]
    for i in xrange(n_theta):
        temp = np.matmul(np.cos(theta_points[i]) * np.cos(phi_points).transpose(), lu_z) \
               + np.matmul(np.cos(theta_points[i]) * np.sin(phi_points).transpose(), ru_z) \
               + np.matmul(np.sin(theta_points[i]) * np.cos(phi_points).transpose(), ld_z) \
               + np.matmul(np.sin(theta_points[i]) * np.sin(phi_points).transpose(), rd_z)
        interp_z[i * n_theta:(i + 1) * n_theta] = temp
    return interp_z