import os
import sys
import numpy as np
import h5py

from utils.data_prep_chordiogram import spherical2cartesian, cartesian2spherical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx



def _rot_z(theta):
    return np.array([ [np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1] ])


def _rot_y(theta):
    return np.array([ [np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)] ])


def rotate_point_cloud(batch_data, mode, rot_type):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
          mode: ['train', 'test']
          rot_type: ['z-z', 'z-so3', 'so3-so3']
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    
    if rot_type not in ['None']:
        idx = 0 if mode is 'train' else 1
        rot_type = rot_type.split('-')[idx]
    else:
        return batch_data

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):

        if rot_type in ['z']:
            # print("rot type={}".format(rot_type))
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = _rot_z(rotation_angle)
        elif rot_type in ['so3']: # rot is 'so3'
            # print("rot type={}".format(rot_type))
            alpha, beta, gamma = np.random.uniform(size=3) * 2 * np.pi
            rotation_matrix = np.dot( np.dot(_rot_z(alpha), _rot_y(beta)), _rot_z(gamma))
        else: # rot_type is None
            rotation_matrix = np.eye(3)

        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_plane0_point_cloud(batch_data, mode, rot_type):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx7 array, original batch of point clouds
          mode: ['train', 'test']
          rot_type: ['z-z', 'z-so3', 'so3-so3']
        Return:
          BxNx7 array, rotated batch of point clouds
    """
    
    idx = 0 if mode is 'train' else 1
    rot_type = rot_type.split('-')[idx]

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):

        if rot_type in ['z']:
            print("mode={}, idx={}, rot type={}".format(mode, idx,rot_type))
            rotation = np.zeros(7)
            rotation[1] = np.random.uniform() * 2 * np.pi
        elif rot_type in ['so3']: # rot is 'so3'
            print("mode={}, idx={}, rot type={}".format(mode, idx,rot_type))
            rotation = np.zeros(7)
            rotation[1] = np.random.uniform() * 2 * np.pi
            rotation[2] = np.random.uniform() * np.pi
        else: # rot_type is None
            rotation = np.zeros(7)

        shape_pc = batch_data[k, ...]
        rotated_shape_pc = shape_pc.reshape((-1, 7)) + rotation.reshape((-1,7))
        # print('shape_pc={}, rotated_shape={}, azimutal={}, elevation={}'.format(shape_pc[0], rotated_shape_pc[0], rotated_shape_pc[0,1], rotated_shape_pc[0,2]))

        x_aux, y_aux, z_aux = spherical2cartesian(rotated_shape_pc[:, 1], rotated_shape_pc[:, 2])
        # print('x_aux={}, y_aux={}, z_aux={}'.format(x_aux[0], y_aux[0], z_aux[0]))
        _, phi, theta = cartesian2spherical(x_aux, y_aux, z_aux)
        # print('phi={}, theta={}'.format(phi[0], theta[0]))

        rotated_shape_pc[:, 1:3] = np.vstack((phi.reshape(1,-1), theta.reshape(1,-1))).T
        # print('rotated_pc={}'.format(rotated_shape_pc[0]))

        rotated_data[k, ...] = rotated_shape_pc

    return rotated_data



def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data

    return jittered_data


def jitter_plane0(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter chords in the plane0 representation. jittering is per point.
        Input:
          BxNx3 array, original batch of chords
        Return:
          BxNx3 array, jittered batch of chords
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data

    # Spherical coordinates correction
    for i in range(3):
        x_aux, y_aux, z_aux = spherical2cartesian(jittered_data[:, :, 2*i + 1], jittered_data[:, :, 2*i + 2])
        _, phi, theta = cartesian2spherical(x_aux, y_aux, z_aux)
        jittered_data[:, :, 2 * i + 1] = phi
        jittered_data[:, :, 2 * i + 2] = theta

    return jittered_data


def jitter_darboux(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter chords in the plane0 representation. jittering is per point.
        Input:
          BxNx3 array, original batch of chords
        Return:
          BxNx3 array, jittered batch of chords
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data

    # Spherical coordinates correction
    x_aux, y_aux, z_aux = spherical2cartesian(jittered_data[:, :, 2], jittered_data[:, :, 3])
    _, phi, theta = cartesian2spherical(x_aux, y_aux, z_aux)
    jittered_data[:, :, 2] = phi
    jittered_data[:, :, 3] = theta

    return jittered_data


def jitter_plane1(batch_data, sigma=0.01, clip=0.05):
    return jitter_point_cloud(batch_data, sigma, clip)


def jitter_plane2(batch_data, sigma=0.01, clip=0.05):
    return jitter_point_cloud(batch_data, sigma, clip)


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
