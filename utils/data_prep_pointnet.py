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


def _sample_points(mesh, num_samples, sampling_method='random'):


    num_sample_faces = min(num_samples, mesh.faces.shape[0])
    if sampling_method == 'area_weighted':
        p = mesh.area_faces / mesh.area
    else:
        p = np.ones(mesh.faces.shape[0]) / mesh.faces.shape[0]
    faces_idx = np.random.choice(mesh.faces.shape[0], size=num_sample_faces, p=p, replace=False)

    sample_faces = mesh.faces[faces_idx, :]

    points = _compute_point_in_triangle(mesh, num_sample_faces, sample_faces)

    return points, num_sample_faces


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


def extract_points(mesh, num_samples=100, sampling_method='uniform', cluster=False):


    chrs, sel = _sample_points(mesh, num_samples, sampling_method=sampling_method)

    '''num_sample_faces = 2 * num_samples
    if sampling_method == 'area_weighted':
        p = mesh.area_faces / mesh.area
    else:
        p = np.ones(mesh.faces.shape[0]) / mesh.faces.shape[0]
    chord_pairs = np.random.choice(mesh.faces.shape[0], size=(num_sample_faces, 2), p=p)



    selected_points = min(num_samples, mesh.vertices.shape[0])
    idx = np.arange(selected_points)
    np.random.shuffle(idx)

    chrs = mesh.vertices[idx, :]

    return chrs, selected_points'''


def create_h5(dataset, file_num, classes, paths, output_path, sampling_method, num_samples=100,
            mode='train', num_augment=1, cluster=False):
    output_filename = os.path.join(output_path,
                                   '{}_points_{}_num_samples_{}_{}{}_augment_{}.h5'.format(dataset, sampling_method,
                                                                                                      num_samples, mode,
                                                                                                      file_num,
                                                                                                      num_augment))
    batch_size = len(paths)
    data = np.zeros((batch_size, num_samples, 3))
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

        chrs, num_selected = _sample_points(mesh, num_samples=num_samples, sampling_method=sampling_method)

        data[i, 0:num_selected, :] = np.nan_to_num(chrs)

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

    dset_folder = args.output_dataset
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
        create_h5(args.dataset, i, train_classes[i * d:(i + 1) * d],
                              all_train_paths[i * d:(i + 1) * d],
                              dset_folder,
                              args.sampling_method, args.num_samples, 'train',
                              args.num_augment, args.cluster)'''

    num_h5_train = int(np.ceil(len(all_train_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(
        delayed(create_h5)(args.dataset, i, train_classes[i * d:(i + 1) * d],
                                       all_train_paths[i * d:(i + 1) * d],
                           dset_folder,
                           args.sampling_method, args.num_samples, 'train',
                           args.num_augment, args.cluster)
        for i in range(num_h5_train))

    num_h5_test = int(np.ceil(len(all_test_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(
        delayed(create_h5)(args.dataset, i, test_classes[i * d:(i + 1) * d],
                                       all_test_paths[i * d:(i + 1) * d],
                           dset_folder,
                           args.sampling_method, args.num_samples, 'test',
                           args.num_augment, args.cluster)
        for i in range(num_h5_test))
