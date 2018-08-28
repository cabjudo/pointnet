# import matplotlib
# matplotlib.use('TkAgg')
import argparse
import glob
import numpy as np
import os
import trimesh
import h5py
from joblib import Parallel, delayed

def cartesian2spherical(x, y, z, r=None):
    if r is None:
        r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)

    return r, theta, phi


def create_chordiogram_h5(file_num, batch_size, classes, paths, output_path, num_samples=100):

    output_filename = os.path.join(output_path, 'ply_data_train{}.h5'.format(file_num))
    data = np.zeros((batch_size, 9900, 7))
    print(data.shape)
    labels = np.zeros((batch_size, 1), dtype=int)

    for i, f in enumerate(paths):
        print('{}: generatin file {}, with {}'.format(i, output_filename, f))
        mesh = trimesh.load(f)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.vertices -= mesh.centroid
        mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

        chr = get_chords(mesh, num_samples=num_samples)
        data[i, :, :] = chr

        path = os.path.normpath(f)
        s = path.split(os.sep)[-3]
        labels[i, 0] = classes[s]

    f = h5py.File(output_filename, 'w')
    f['data'] = data
    f['label'] = labels
    f.close()


def get_chords(mesh, num_samples=100):

    num_sample_faces = num_samples
    num_chords = num_sample_faces*(num_sample_faces - 1)
    chrs = np.zeros((num_chords, 7))

    if mesh.faces.shape[0] < num_sample_faces:
        sample_faces = mesh.faces
        num_sample_faces = mesh.faces.shape[0]
        num_chords = num_sample_faces * (num_sample_faces - 1)
        normals = mesh.face_normals
    else:
        idx = np.arange(mesh.faces.shape[0])
        np.random.shuffle(idx)
        sample_faces = mesh.faces[idx[:num_sample_faces], :]
        normals = mesh.face_normals[idx[:num_sample_faces], :]

    point1 = mesh.vertices[sample_faces[:, 0], :]
    point2 = mesh.vertices[sample_faces[:, 1], :]
    point3 = mesh.vertices[sample_faces[:, 2], :]

    c = (point1 + point2 + point3) / 3.0

    m_x = np.dot(c[:, 0][:, None], np.ones((1, num_sample_faces))) - np.dot(np.ones((num_sample_faces, 1)), c[:, 0][None])
    m_y = np.dot(c[:, 1][:, None], np.ones((1, num_sample_faces))) - np.dot(np.ones((num_sample_faces, 1)), c[:, 1][None])
    m_z = np.dot(c[:, 2][:, None], np.ones((1, num_sample_faces))) - np.dot(np.ones((num_sample_faces, 1)), c[:, 2][None])
    m = np.stack((m_x, m_y, m_z), axis=2)

    n_x = np.dot(np.ones((num_sample_faces, 1)), normals[:, 0][None])
    n_y = np.dot(np.ones((num_sample_faces, 1)), normals[:, 1][None])
    n_z = np.dot(np.ones((num_sample_faces, 1)), normals[:, 2][None])
    n = np.stack((n_x, n_y, n_z), axis=2)

    r = np.sqrt(np.sum(m**2, axis=2))

    idx = np.arange(num_sample_faces), np.arange(num_sample_faces)
    idx = idx[0] * sample_faces.shape[0] + idx[1]

    # length of the chord
    chrs[0:num_chords, 0] = np.delete(r.ravel(), idx)

    # chord orientation
    r, theta, phi = cartesian2spherical(m_x, m_y, m_z, r=r)
    theta = np.delete(theta.ravel(), idx)
    phi = np.delete(phi.ravel(), idx)
    chrs[0:num_chords, 1:3] = np.vstack((theta, phi)).T

    # chord normalized normal on p
    r, theta, phi = cartesian2spherical(n_x, n_y, n_z)
    theta = np.delete(theta.ravel(), idx)
    phi = np.delete(phi.ravel(), idx)
    chrs[0:num_chords, 3:5] = np.vstack((theta, phi)).T - chrs[0:num_chords, 1:3]

    # chord normalized normal on q
    r, theta, phi = cartesian2spherical(n_x.T, n_y.T, n_z.T, r=r)
    theta = np.delete(theta.ravel(), idx)
    phi = np.delete(phi.ravel(), idx)
    chrs[0:num_chords, 5:7] = np.vstack((theta, phi)).T - chrs[0:num_chords, 1:3]

    return chrs

def read_checkpoint(output_path):
    checkpoints_folder = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    file = open(os.path.join(checkpoints_folder, '{}.txt').format(filenumber), 'w')
    file.write('done.')
    file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--input', help='Input Modelnet folder', dest='input_dataset', default='.')
    parser.add_argument('-b', '--batch_size', help="Batch size for each h5 file", dest="batch_size", type=int)
    parser.add_argument('-m', '--method', help="Method to sample the point cloud", dest="sampling_method", choices=['random'], default='random')
    parser.add_argument('-o', '--output', help="Output folder", dest='output_dataset')
    parser.add_argument('-t', '--mode', help="Training or testing mode", dest='mode', default='train')
    args = parser.parse_args()

    if not os.path.exists(args.output_dataset):
        os.makedirs(args.output_dataset)

    all_train_paths = np.array(glob.glob(os.path.join(args.input_dataset, '*/train/*.off')))
    all_test_paths = np.array(glob.glob(os.path.join(args.input_dataset, '*/test/*.off')))
    all_paths = {'train': np.random.shuffle(all_train_paths),
                 'test': np.random.shuffle(all_test_paths)}

    # generate de classes label
    i = 0
    classes = {}
    for f in all_train_paths:
        path = os.path.normpath(f)
        s = path.split(os.sep)[-3]
        if s not in classes:
            classes[s] = i
            i += 1

    num_h5 = int(np.ceil(len(all_train_paths) / args.batch_size))
    d = args.batch_size
    Parallel(n_jobs=-1, timeout=600)(delayed(create_chordiogram_h5)(i, d, all_train_paths[i * d:(i + 1) * d], args.output_dataset) for i in range(num_h5))

    #for i in range(num_h5):
    #    create_chordiogram_h5(i, d, classes, all_train_paths[i * d:(i + 1) * d], args.output_dataset)

