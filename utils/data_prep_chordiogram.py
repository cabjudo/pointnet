# import matplotlib
# matplotlib.use('TkAgg')
import argparse
import glob
import numpy as np
import os
import pandas as pd
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

    phi = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    theta = np.arccos(z / r)
    theta = np.nan_to_num(theta)

    return r, phi, theta


def spherical2cartesian(phi, theta, r=None):
    if r is None:
        r = 1.0

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def axis_angle_to_rotmat(z):
    ''' Converts from axis angle to a rotation matrix
    Input:
    z - (3,) axis angle representation
    '''

    # theta=angle, z_n = unit norm direction
    theta = np.linalg.norm(z, axis=1)[:, None]
    z_n = z / theta
    # print("theta={}, ||z_n||={}".format(theta, np.linalg.norm(z_n)))

    # infinitesimal rotation from z_n
    omega = np.zeros((z.shape[0], 3, 3))
    # skew symmetric representation from z_n
    omega[:, 1, 0] = z_n[:, 2]
    omega[:, 2, 0] = -z_n[:, 1]
    omega[:, 2, 1] = z_n[:, 0]
    # print("omega={}".format(omega))
    omega = omega - np.transpose(omega, axes=[0, 2, 1])
    # print("omega={}".format(omega))

    # rodrigues formula
    R = np.eye(3) + np.sin(theta) * omega + (1 - np.cos(theta)) * np.dot(omega, omega)

    return R, theta, z_n


def _sample_chords(mesh, num_samples, sampling_method='random'):
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


def plane0_chord_representation(mesh, num_samples=100, sampling_method='uniform', cluster=False):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 7))

    # length of the chord
    chrs[:, 0] = r

    # chord orientation
    r, phi, theta = cartesian2spherical(m[:, 0], m[:, 1], m[:, 2], r=r)
    chrs[:, 1:3] = np.vstack((phi, theta)).T

    # chord normalized normal on p
    r_p, phi_p, theta_p = cartesian2spherical(n_p[:, 0], n_p[:, 1], n_p[:, 2])
    x_aux, y_aux, z_aux = spherical2cartesian(phi_p - chrs[:, 1], theta_p - chrs[:, 2])
    r_p_norm, phi_p_norm, theta_p_norm = cartesian2spherical(x_aux, y_aux, z_aux)
    chrs[:, 3:5] = np.vstack((phi_p_norm, theta_p_norm)).T

    # chord normalized normal on q
    r_q, phi_q, theta_q = cartesian2spherical(n_q[:, 0], n_q[:, 1], n_q[:, 2])
    x_aux, y_aux, z_aux = spherical2cartesian(phi_q - chrs[:, 1], theta_q - chrs[:, 2])
    r_q_norm, phi_q_norm, theta_q_norm = cartesian2spherical(x_aux, y_aux, z_aux)
    chrs[:, 5:7] = np.vstack((phi_q_norm, theta_q_norm)).T

    return chrs


def plane1_chord_representation(mesh, num_samples=100, sampling_method='uniform', cluster=False):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 3))

    # length of the chord
    chrs[:, 0] = r

    # Angle of normal on p with respect to the chord
    # NOTE: normals are unit vectors
    ori_p = np.sum(m * n_p, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[:, 1] = ori_p

    # Angle of the normal on q with respect to the plane generated by the normal on p and the chord
    # https://www.vitutor.com/geometry/distance/line_plane.html
    # NOTE: normals are unit vectors
    plane1 = np.cross(n_p, m)
    ori_q = np.arcsin(np.sum(plane1 * n_q, axis=1) / (np.linalg.norm(plane1, axis=1)))
    chrs[:, 2] = ori_q

    return chrs


def plane2_chord_representation(mesh, num_samples=100, sampling_method='uniform', cluster=False):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 4))

    # length of the chord
    chrs[:, 0] = r

    # Orientation of the normal at p wrt the chord
    # NOTE: normals are unit vectors
    ori_p = np.sum(m * n_p, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[:, 1] = ori_p

    # Orientation of the normal at q wrt the chord
    ori_q = np.sum(- m * n_q, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[:, 2] = ori_q

    # Angle between the two planes formed by the normal and the plane
    plane_p = np.cross(n_p, m)
    plane_q = np.cross(n_q, m)
    plane_angle = np.sum(plane_p * plane_q, axis=1) / (
        np.linalg.norm(plane_p, axis=1) * np.linalg.norm(plane_q, axis=1))
    chrs[:, 3] = plane_angle

    return chrs


def darboux_chord_representation(mesh, num_samples=100, sampling_method='uniform', cluster=False):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 4))

    # Computes the orthonormal basis if the Darboux frame
    u = n_p  # Normals are already unit vectors
    v = np.cross(u, -m)
    v /= np.linalg.norm(v, axis=1)[:, None]
    w = np.cross(u, v)
    w /= np.linalg.norm(w, axis=1)[:, None]

    # length of the chord
    chrs[:, 0] = r

    # Orientation of the normal at p wrt the chord
    ori_p = np.sum(-m * u, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[:, 1] = ori_p

    # Spherical coordinates (theta and phi) of the normal at q according to the coordinate system
    # defined by the Darboux frame
    basis = np.stack((u, v, w), axis=2)
    basis_inverse = np.transpose(basis, axes=[0, 2, 1])
    n_q_transformed = np.squeeze(np.matmul(basis_inverse, n_q[:, :, None]))
    _, phi, theta = cartesian2spherical(n_q_transformed[:, 0], n_q_transformed[:, 1], n_q_transformed[:, 2])

    chrs[:, 2:4] = np.vstack((phi, theta)).T

    return chrs


def darboux_aug_chord_representation(mesh, num_samples=100, sampling_method='uniform', cluster=False):
    m, n_p, n_q, num_chords = _sample_chords(mesh, num_samples, sampling_method)
    r = np.sqrt(np.sum(m ** 2, axis=1))

    chrs = np.zeros((num_samples, 5))

    # Computes the orthonormal basis if the Darboux frame
    u = n_p  # Normals are already unit vectors
    v = np.cross(u, -m)
    v /= np.linalg.norm(v, axis=1)[:, None]
    w = np.cross(u, v)
    w /= np.linalg.norm(w, axis=1)[:, None]

    # length of the chord
    chrs[:, 0] = r

    # Orientation of the normal at p wrt the chord
    ori_p = np.sum(-m * u, axis=1) / (np.linalg.norm(m, axis=1))
    chrs[:, 1] = ori_p

    # Spherical coordinates (theta and phi) of the normal at q according to the coordinate system
    # defined by the Darboux frame
    basis = np.stack((u, v, w), axis=2)
    basis_inverse = np.transpose(basis, axes=[0, 2, 1])
    n_q_transformed = np.squeeze(np.matmul(basis_inverse, n_q[:, :, None]))
    _, phi, theta = cartesian2spherical(n_q_transformed[:, 0], n_q_transformed[:, 1], n_q_transformed[:, 2])
    chrs[:, 2:4] = np.vstack((phi, theta)).T

    # Angle between the two planes formed by the normal and the plane
    plane_p = np.cross(n_p, m)
    plane_q = np.cross(n_q, m)
    plane_angle = np.sum(plane_p * plane_q, axis=1) / (
        np.linalg.norm(plane_p, axis=1) * np.linalg.norm(plane_q, axis=1))
    chrs[:, 4] = plane_angle

    return chrs


METHODS = {
    'plane0': {'num_features': 7, 'function': plane0_chord_representation},
    'plane1': {'num_features': 3, 'function': plane1_chord_representation},
    'plane2': {'num_features': 4, 'function': plane2_chord_representation},
    'darboux': {'num_features': 4, 'function': darboux_chord_representation},
    'darboux_aug': {'num_features': 5, 'function': darboux_aug_chord_representation}
}


def create_chordiogram_h5(file_num, classes, paths, output_path, sampling_method, num_samples=100,
                          chord_type='plane1', mode='train', num_augment=1, cluster=False):
    output_filename = os.path.join(output_path,
                                   'modelnet40_chords_{}_{}_num_samples_{}_{}{}_augment_{}.h5'.format(sampling_method,
                                                                                                      chord_type,
                                                                                                      num_samples, mode,
                                                                                                      file_num,
                                                                                                      num_augment))
    batch_size = len(paths)
    data = np.zeros((batch_size, num_samples, METHODS[chord_type]['num_features']))
    labels = np.zeros((batch_size, 1), dtype=int)
    labels[:, 0] = classes
    fnames = []

    for i, f in enumerate(paths):

        try:
            mesh = trimesh.load(f)
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.vertices -= mesh.centroid
            mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()
            print('{}: generatin file {}, with {}'.format(i, output_filename, f))
        except:
            print('Error processing: {}'.format(f))
            continue

        chr = METHODS[chord_type]['function'](mesh, num_samples=num_samples, sampling_method=sampling_method,
                                              cluster=cluster)
        data[i, :, :] = np.nan_to_num(chr)

        path = os.path.normpath(f)
        s = path.split(os.sep)[-1]

        fnames.append(s)

    f = h5py.File(output_filename, 'w')
    f['data'] = data
    f['label'] = labels
    f['fnames'] = np.string_(fnames)
    f.close()


def get_modelnet_files(input_dataset_folder, num_augment=1):
    all_train_paths = np.repeat(np.array(glob.glob(os.path.join(input_dataset_folder, '*/train/*.off'))),
                                num_augment)
    all_test_paths = np.array(glob.glob(os.path.join(input_dataset_folder, '*/test/*.off')))
    np.random.shuffle(all_train_paths)
    np.random.shuffle(all_test_paths)
    # all_paths = all_train_paths if args.mode == 'train' else all_test_paths
    # generate de classes label

    train_classes = np.array([os.path.normpath(f).split(os.sep)[-3] for i, f in enumerate(all_train_paths)])
    idx = np.argsort(train_classes)
    l, _ = pd.factorize(train_classes[idx])
    train_classes[idx] = l

    test_classes = np.array([os.path.normpath(f).split(os.sep)[-3] for i, f in enumerate(all_test_paths)])
    idx = np.argsort(test_classes)
    l, _ = pd.factorize(test_classes[idx])
    test_classes[idx] = l

    return all_train_paths, all_test_paths, train_classes, test_classes


def get_shrec17_files(input_dataset_folder, num_augment=1):
    '''

    Args:
        input_dataset_folder:
        num_augment:

    Returns:

    '''

    # gets all the training/testin filenames and sort them by object id
    all_train_paths = np.repeat(np.array(glob.glob(os.path.join(input_dataset_folder, 'train_normal/*.obj'))),
                                num_augment)

    all_test_paths = np.array(glob.glob(os.path.join(input_dataset_folder, 'val_normal/*.obj')))
    idx = np.argsort([int(f.split(os.sep)[-1].split('.')[0]) for f in all_test_paths])
    all_test_paths = all_test_paths[idx]

    labels_file = pd.read_csv(os.path.join(input_dataset_folder, 'train.csv'))
    all_train_paths, train_classes = _compute_shrec17_labels(all_train_paths, labels_file)

    labels_file = pd.read_csv(os.path.join(input_dataset_folder, 'val.csv'))
    all_test_paths, test_classes = _compute_shrec17_labels(all_test_paths, labels_file)

    return all_train_paths, all_test_paths, train_classes, test_classes


def _compute_shrec17_labels(all_paths, labels_file):
    assert all_paths.shape[0] == labels_file.values.shape[0], \
        'The .csv with the labels and the number of files in the folder must match.'

    idx = np.argsort([int(f.split(os.sep)[-1].split('.')[0]) for f in all_paths])
    all_paths = all_paths[idx]

    # Replace the class id with consecutive values (0 to number of classes)
    data = labels_file.values[:, 0:2]
    idx = np.argsort(data[:, 1])
    l, _ = pd.factorize(data[idx, 1])
    data[idx, 1] = l
    # Sort labels by object id (now filenames and labels match)
    idx = np.argsort(data[:, 0])
    data = data[idx, :]
    # shuffle labels and filenames at the same time
    idx = np.array(range(data.shape[0]))
    np.random.shuffle(idx)
    data = data[idx, :]
    all_paths = all_paths[idx]
    labels = data[:, 1]

    return all_paths, labels


if __name__ == '__main__':

    sampling_methods = ['random', 'area_weighted']
    chord_type = ["plane0", "plane1", "plane2", "original", "darboux", "darboux_aug"]
    all_datasets = ['modelnet40', 'shrec17']

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help='Dataset', dest='dataset', default='.', choices=all_datasets)
    parser.add_argument('-i', '--input_folder', help='Input dataset folder folder', dest='input_dataset_folder',
                        default='.')
    parser.add_argument('-b', '--batch_size', help="Batch size for each h5 file", dest="batch_size", type=int,
                        default=2048)
    parser.add_argument('-m', '--method', help="Method to sample the point cloud", dest="sampling_method",
                        choices=sampling_methods, default='random')
    parser.add_argument('-o', '--output', help="Output folder", dest='output_dataset')
    parser.add_argument('-c', '--chord_type', choices=chord_type, help="Chord feature type", dest='chord_type',
                        default='plane2')
    parser.add_argument('-n', '--num_samples', help="num of random samples", dest='num_samples', default=2048, type=int)
    parser.add_argument('-a', '--augment', help="Num. of times each shape will be sampled", dest='num_augment',
                        default=1, type=int)
    parser.add_argument('-k', '--cluster',
                        help="Calculates of the chords in a shape and cluster them in to 'num_samples' clusters",
                        dest='cluster',
                        default=False, type=bool)
    args = parser.parse_args()

    assert (args.dataset != 'shrec17' or args.num_augment == 1), 'Augmentation for Shrec17 not supported.'

    if not os.path.exists(args.output_dataset):
        os.makedirs(args.output_dataset)

    dset_folder = os.path.expanduser(os.path.join(args.output_dataset, args.chord_type))
    if not os.path.exists(dset_folder):
        os.makedirs(dset_folder)

    if args.dataset in ['modelnet40']:
        all_train_paths, all_test_paths, train_classes, test_classes = get_modelnet_files(
            args.input_dataset_folder, args.num_augment)
    elif args.dataset == 'shrec17':
        all_train_paths, all_test_paths, train_classes, test_classes = get_shrec17_files(
            args.input_dataset_folder, args.num_augment)

    '''num_h5_train = int(np.ceil(len(all_train_paths) / args.batch_size))
    d = args.batch_size
    for i in range(num_h5_train):
        create_chordiogram_h5(i, train_classes[i * d:(i + 1) * d],
                              all_train_paths[i * d:(i + 1) * d],
                              dset_folder,
                              args.sampling_method, args.num_samples, args.chord_type, 'train',
                              args.num_augment, args.cluster)'''

    num_h5_train = int(np.ceil(len(all_train_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(
        delayed(create_chordiogram_h5)(i, train_classes[i * d:(i + 1) * d],
                                       all_train_paths[i * d:(i + 1) * d],
                                       dset_folder,
                                       args.sampling_method, args.num_samples, args.chord_type, 'train',
                                       args.num_augment, args.cluster)
        for i in range(num_h5_train))

    num_h5_test = int(np.ceil(len(all_test_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(
        delayed(create_chordiogram_h5)(i, test_classes[i * d:(i + 1) * d],
                                       all_test_paths[i * d:(i + 1) * d],
                                       dset_folder,
                                       args.sampling_method, args.num_samples, args.chord_type, 'test',
                                       args.num_augment, args.cluster)
        for i in range(num_h5_test))
