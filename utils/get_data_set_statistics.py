# import matplotlib
# matplotlib.use('TkAgg')
import argparse
import glob
import numpy as np
import os
import trimesh
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--input', help='Input Modelnet folder', dest='input_dataset', default='.')
    parser.add_argument('-o', '--output', help="Output filename", dest='output_filename')
    args = parser.parse_args()

    all_train_paths = glob.glob(os.path.join(args.input_dataset, '*/train/*.off'))
    all_test_paths = glob.glob(os.path.join(args.input_dataset, '*/test/*.off'))

    all_paths = all_train_paths
    all_paths.extend(all_test_paths)

    with open(args.output_filename, 'w') as csvfile:

        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Filename', 'Mode', 'Filesize', 'Num. Triangles', 'Num. Points'])

        for fn in all_paths:

            print('Working on {}.'.format(fn))

            path_parts = os.path.normpath(fn).split(os.sep)

            # filename
            filename = os.path.join(*path_parts[-3:])

            # Mode
            mode = path_parts[-2]

            #filesize
            filesize = os.path.getsize(fn)

            mesh = trimesh.load(fn)

            #num triangles
            num_triangles = mesh.faces.shape[0]

            #num points
            num_points = mesh.vertices.shape[0]

            filewriter.writerow([filename, mode, filesize, num_triangles, num_points])

