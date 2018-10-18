# import matplotlib
# matplotlib.use('TkAgg')
import argparse
import glob
import numpy as np
import os
import trimesh
import h5py
from joblib import Parallel, delayed


def _get_rotation(angles):
    alpha, beta, gamma = angles
    R = np.dot(np.dot(_rot_z(alpha), _rot_y(beta)), _rot_z(gamma))

    return R


def _rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def _rot_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rotate_point_cloud(mesh):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
    """
    T = np.zeros((4, 4))
    angles = [np.random.uniform() * 2 * np.pi for i in range(3)]
    T[:3, :3] = _get_rotation(*angles)
    T[-1, -1] = 1

    mesh.apply_transform(T)
    return mesh


def cartesian2spherical(x, y, z, r=None):
    if r is None:
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    phi = np.nan_to_num(phi)

    return r, theta, phi


def _uniform_sample_chords(mesh, num_samples):


    num_sample_faces = num_samples
    if mesh.faces.shape[0] < num_sample_faces:
        sample_faces = mesh.faces
        num_sample_faces = mesh.faces.shape[0]
        normals = -mesh.face_normals
    else:
        idx = np.arange(mesh.faces.shape[0])
        np.random.shuffle(idx)
        sample_faces = mesh.faces[idx[:num_sample_faces], :]
        normals = -mesh.face_normals[idx[:num_sample_faces], :]
    point1 = mesh.vertices[sample_faces[:, 0], :]
    point2 = mesh.vertices[sample_faces[:, 1], :]
    point3 = mesh.vertices[sample_faces[:, 2], :]
    c = (point1 + point2 + point3) / 3.0
    m_x = np.dot(c[:, 0][:, None], np.ones((1, num_sample_faces))) - np.dot(np.ones((num_sample_faces, 1)),
                                                                            c[:, 0][None])
    m_y = np.dot(c[:, 1][:, None], np.ones((1, num_sample_faces))) - np.dot(np.ones((num_sample_faces, 1)),
                                                                            c[:, 1][None])
    m_z = np.dot(c[:, 2][:, None], np.ones((1, num_sample_faces))) - np.dot(np.ones((num_sample_faces, 1)),
                                                                            c[:, 2][None])
    m = np.stack((m_x, m_y, m_z), axis=2)
    n_x = np.dot(np.ones((num_sample_faces, 1)), normals[:, 0][None])
    n_y = np.dot(np.ones((num_sample_faces, 1)), normals[:, 1][None])
    n_z = np.dot(np.ones((num_sample_faces, 1)), normals[:, 2][None])
    n_p = np.stack((n_x, n_y, n_z), axis=2)
    n_q = np.transpose(n_p, (1, 0, 2))

    return m, n_p, n_q, num_sample_faces


def _sample_chords(mesh, num_samples, sampling_method='random'):

    if num_samples > mesh.faces.shape[0]**2:

        num_sample_faces = mesh.faces.shape[0] ** 2
        idx = np.arange(mesh.faces.shape[0])
        chord_pairs = np.transpose([np.tile(idx, idx.size), np.repeat(idx, idx.size)])
        aux = np.arange(num_sample_faces)
        np.random.shuffle(aux)
        chord_pairs = chord_pairs[aux, :]

    else:

        num_sample_faces = num_samples
        if sampling_method == 'area_weighted':
            p = mesh.area_faces / mesh.area
        else:
            p = np.ones(mesh.faces.shape[0]) / mesh.faces.shape[0]
        chord_pairs = np.random.choice(mesh.faces.shape[0], size=(num_samples, 2), p=p)

    sample_faces_left = mesh.faces[chord_pairs[:, 0], :]
    sample_faces_right = mesh.faces[chord_pairs[:, 1], :]
    n_p = -mesh.face_normals[chord_pairs[:, 0], :]
    n_q = -mesh.face_normals[chord_pairs[:, 1], :]

    c_left = _compute_point_in_triangle(mesh, num_sample_faces, sample_faces_left)
    c_right = _compute_point_in_triangle(mesh, num_sample_faces, sample_faces_right)
    m = c_left - c_right

    return m, n_p, n_q, num_sample_faces


def _compute_point_in_triangle(mesh, num_sample_faces, face_indexes):

    aux = np.random.rand(num_sample_faces, 3)
    aux /= np.dot(aux.sum(axis=1)[:, None], np.ones((1, 3)))
    lambdas = np.stack((aux, aux, aux), axis=1)
    points1_left = mesh.vertices[face_indexes[:, 0], :]
    points2_left = mesh.vertices[face_indexes[:, 1], :]
    points3_left = mesh.vertices[face_indexes[:, 2], :]
    k = np.stack((points1_left, points2_left, points3_left), axis=2) * lambdas
    points = np.sum(k, axis=2)
    return points


def get_chords1(mesh, num_samples=100, sampling_method='uniform'):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 7))

    # length of the chord
    chrs[0:num_chords, 0] = r

    # chord orientation
    r, theta, phi = cartesian2spherical(m[:, 0], m[:, 1], m[:, 2], r=r)
    chrs[0:num_chords, 1:3] = np.vstack((theta, phi)).T

    # chord normalized normal on p
    r, theta, phi = cartesian2spherical(n_p[:, 0], n_p[:, 1], n_p[:, 2])
    chrs[0:num_chords, 3:5] = np.vstack((theta, phi)).T - chrs[0:num_chords, 1:3]

    # chord normalized normal on q
    r, theta, phi = cartesian2spherical(n_q[:, 0], n_q[:, 1], n_q[:, 2], r=r)
    chrs[0:num_chords, 5:7] = np.vstack((theta, phi)).T - chrs[0:num_chords, 1:3]

    return chrs


def get_chords2(mesh, num_samples=100, sampling_method='uniform'):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 3))

    # length of the chord
    chrs[0:num_chords, 0] = r

    # Angle of normal on p with respect to the chord
    # NOTE: normals are unit vectors
    ori_p = np.sum(m * n_p, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[0:num_chords, 1] = ori_p

    # Angle of the normal on q with respect to the plane generated by the normal on p and the chord
    # https://www.vitutor.com/geometry/distance/line_plane.html
    # NOTE: normals are unit vectors
    plane1 = np.cross(n_p, m)
    ori_q = np.arcsin(np.sum(plane1 * n_q, axis=1) / (np.linalg.norm(plane1, axis=1)))
    chrs[0:num_chords, 2] = ori_q

    return chrs


def get_chords3(mesh, num_samples=100, sampling_method='uniform'):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 4))

    # length of the chord
    chrs[0:num_chords, 0] = r

    # Orientation of the normal at p wrt the chord
    # NOTE: normals are unit vectors
    ori_p = np.sum(m * n_p, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[0:num_chords, 1] = ori_p

    # Orientation of the normal at q wrt the chord
    ori_q = np.sum(- m * n_q, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[0:num_chords, 2] = ori_q

    # Angle between the two planes formed by the normal and the plane
    plane_p = np.cross(n_p, m)
    plane_q = np.cross(n_q, m)
    plane_angle = np.sum(plane_p * plane_q, axis=1) / (
        np.linalg.norm(plane_p, axis=1) * np.linalg.norm(plane_q, axis=1))
    chrs[0:num_chords, 3] = plane_angle

    return chrs


METHODS = {1: {'name': '6_angles', 'num_features': 7, 'function': get_chords1},
           2: {'name': '2_angles', 'num_features': 3, 'function': get_chords2},
           3: {'name': 'angle_bt_planes', 'num_features': 4, 'function': get_chords3}
           }

SAMPLING_METHODS = {'random': _uniform_sample_chords,
                    'area_weighted': _sample_chords
                    }


def create_chordiogram_h5(file_num, batch_size, classes, paths, output_path, sampling_method, num_samples=100,
                          chord_type=1,
                          mode='train'):
    output_filename = os.path.join(output_path,
                                   'modelnet40_chords_{}_{}_num_samples_{}_{}{}.h5'.format(sampling_method, METHODS[chord_type]['name'],
                                                                                       num_samples, mode, file_num))
    data = np.zeros((batch_size, num_samples, METHODS[chord_type]['num_features']))
    labels = np.zeros((batch_size, 1), dtype=int)

    for i, f in enumerate(paths):
        print('{}: generatin file {}, with {}'.format(i, output_filename, f))
        mesh = trimesh.load(f)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh = trimesh.permutate.Permutator().permutate(mesh)
        mesh = trimesh.permutate.Permutator(mesh)
        mesh.vertices -= mesh.centroid
        mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

        chr = METHODS[chord_type]['function'](mesh, num_samples=num_samples, sampling_method=sampling_method)
        data[i, :, :] = np.nan_to_num(chr)

        path = os.path.normpath(f)
        s = path.split(os.sep)[-3]
        labels[i, 0] = classes[s]

    f = h5py.File(output_filename, 'w')
    f['data'] = data
    f['label'] = labels
    f.close()


def read_checkpoint(output_path):
    checkpoints_folder = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    file = open(os.path.join(checkpoints_folder, '{}.txt').format(filenumber), 'w')
    file.write('done.')
    file.close()


if __name__ == '__main__':

    sampling_methods = ['random', 'area_weighted']

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--input', help='Input Modelnet folder', dest='input_dataset', default='.')
    parser.add_argument('-b', '--batch_size', help="Batch size for each h5 file", dest="batch_size", type=int)
    parser.add_argument('-m', '--method', help="Method to sample the point cloud", dest="sampling_method",
                        choices=sampling_methods, default='random')
    parser.add_argument('-o', '--output', help="Output folder", dest='output_dataset')
    parser.add_argument('-c', '--chord_type', help="Chord feature type", dest='chord_type', default=1, type=int)
    parser.add_argument('-n', '--num_samples', help="num of random samples", dest='num_samples', default=10000)
    args = parser.parse_args()

    if not os.path.exists(args.output_dataset):
        os.makedirs(args.output_dataset)

    all_train_paths = np.array(glob.glob(os.path.join(args.input_dataset, '*/train/*.off')))
    all_test_paths = np.array(glob.glob(os.path.join(args.input_dataset, '*/test/*.off')))
    np.random.shuffle(all_train_paths)
    np.random.shuffle(all_test_paths)

    # all_paths = all_train_paths if args.mode == 'train' else all_test_paths

    # generate de classes label
    i = 0
    classes = {}
    for f in all_train_paths:
        path = os.path.normpath(f)
        s = path.split(os.sep)[-3]
        if s not in classes:
            classes[s] = i
            i += 1

    num_h5_train = int(np.ceil(len(all_train_paths) / args.batch_size))
    d = args.batch_size
    for i in range(num_h5_train):
       create_chordiogram_h5(i, d, classes, all_train_paths[i * d:(i + 1) * d], args.output_dataset,
                             args.sampling_method, args.num_samples, args.chord_type, 'train')

    num_h5_train = int(np.ceil(len(all_train_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(
        delayed(create_chordiogram_h5)(i, d, classes, all_train_paths[i * d:(i + 1) * d], args.output_dataset,
                                       args.sampling_method, args.num_samples, args.chord_type, 'train')
        for i in range(num_h5_train))

    num_h5_test = int(np.ceil(len(all_test_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(
        delayed(create_chordiogram_h5)(i, d, classes, all_test_paths[i * d:(i + 1) * d], args.output_dataset,
                                       args.sampling_method, args.num_samples, args.chord_type, 'test')
        for i in range(num_h5_test))

