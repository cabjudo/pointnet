import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score
import provider
import subprocess
# import pc_util


model_choices = ["pointnet_cls", "pointnet_cls_basic", "pointnet_no3trans", "pointnet_notrans"]
dataset_choices = ["plane0", "plane1", "plane2", "original", "darboux"]
train_test = ["z-z", "z-so3", "so3-so3"]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', choices=model_choices, help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--dataset', default='plane1', choices=dataset_choices, help='Dataset: chordiogram representation [default: plane11]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--train_test', default="z-z", help='Decay rate for lr decay [default: z-z]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
# parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
# parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
# parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
# FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
TRAIN_TEST = FLAGS.train_test
DUMP_DIR = FLAGS.dump_dir
LOG_DIR = FLAGS.log_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
# TRAIN_FILES = provider.getDataFiles( \
#     os.path.join(BASE_DIR, '/NAS/data/christine/modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = provider.getDataFiles(\
#     os.path.join(BASE_DIR, '/NAS/data/christine/modelnet40_ply_hdf5_2048/test_files.txt'))

DatasetPath = {
    "plane0": {
        "train": os.path.join(BASE_DIR, '/NAS/data/diego/chords_dataset/plane0/train_files.txt'),
        "test": os.path.join(BASE_DIR, '/NAS/data/diego/chords_dataset/plane0/test_files.txt'),
        "num_chord_features": 7,
    },
    "plane1": {
        "train": os.path.join(BASE_DIR, '/NAS/data/diego/chords_dataset/plane1/train_files.txt'),
        "test": os.path.join(BASE_DIR, '/NAS/data/diego/chords_dataset/plane1/test_files.txt'),
        "num_chord_features": 3,
    },
    "plane2": {
        "train": os.path.join(BASE_DIR, '/NAS/data/diego/chords_dataset/plane2/train_files.txt'),
        "test": os.path.join(BASE_DIR, '/NAS/data/diego/chords_dataset/plane2/test_files.txt'),
        "num_chord_features": 4,
    },
    "original": {
        "train": os.path.join(BASE_DIR, '/NAS/data/christine/modelnet40_ply_hdf5_2048/train_files.txt'),
        "test": os.path.join(BASE_DIR, '/NAS/data/christine/modelnet40_ply_hdf5_2048/test_files.txt'),
        "num_chord_features": 3,
    },
    "darboux": {
        "train": os.path.join(BASE_DIR, '/Users/dipaco/Documents/Datasets/chord_point_net/darboux/train_files.txt'),
        "test": os.path.join(BASE_DIR, '/Users/dipaco/Documents/Datasets/chord_point_net/darboux/test_files.txt'),
        "num_chord_features": 4,
    },
}

DSET_INFO = DatasetPath[FLAGS.dataset]
#TRAIN_FILES = provider.getDataFiles( \
#    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
#TEST_FILES = provider.getDataFiles(\
#    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
TRAIN_FILES = provider.getDataFiles(DSET_INFO['train'])
    # os.path.join(BASE_DIR, '../../data/chords_dataset/train_files_2_angles.txt'))

TEST_FILES = provider.getDataFiles(DSET_INFO['test'])
    # os.path.join(BASE_DIR, '../../data/chords_dataset/test_files_2_angles.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def retrieval():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, DSET_INFO['num_chord_features'])
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points, feature_map = MODEL.get_model(pointclouds_pl, is_training_pl, input_dims=DSET_INFO['num_chord_features'], return_feature_map=True)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'feature_map': feature_map,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    retrieval_one_epoch(sess, ops)

   
def retrieval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    all_descriptors = np.array([[]])
    scores = np.array([[]])
    labels = np.array([[]])
    fnames = np.array([])
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label, current_fnames = provider.loadDataFile(TEST_FILES[fn], return_fnames=True)
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            if FLAGS.dataset in ["original"]:
                rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :], 'test', TRAIN_TEST)
            elif FLAGS.dataset in ["plane0"]:
                rotated_data = provider.rotate_plane0_point_cloud(current_data[start_idx:end_idx, :, :], 'test', TRAIN_TEST)
            else:
                rotated_data = current_data[start_idx:end_idx, :, :]

            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val, feature_map = sess.run([ops['loss'], ops['pred'], ops['feature_map']],
                                      feed_dict=feed_dict)

            if all_descriptors.size > 0:
                all_descriptors = np.vstack((all_descriptors, feature_map))
                scores = np.vstack((scores, pred_val))
                labels = np.hstack((labels, current_label[start_idx:end_idx]))
                fnames = np.hstack((fnames, current_fnames[start_idx:end_idx]))
            else:
                all_descriptors = feature_map
                scores = pred_val
                labels = current_label[start_idx:end_idx]
                fnames = current_fnames[start_idx:end_idx]

    fname = MODEL_PATH + '_dists.npy'
    dists = compute_and_save_descriptors_dists(all_descriptors, fname)

    thresh = search_thresholds(dists, labels)

    make_shrec17_output_thresh(all_descriptors, scores, fnames, MODEL_PATH,
                               distance='cosine', dists=dists, thresh=thresh)

    res = eval_shrec17_output(os.path.split(LOG_DIR)[0])
    #print(modeldir, datadir, ckpt)
    print(res.head(1))
    print(res.tail(1))


def compute_and_save_descriptors_dists(descriptors, filename):
    """ Save descriptors and pairwise distances. """

    if not os.path.exists(filename):
        dist = squareform(pdist(descriptors, 'cosine'))
        np.save(filename, dist)
    else:
        dist = np.load(filename)

    return dist


def make_shrec17_output_thresh(descriptors, scores, fnames, outdir,
                               distance='cosine', dists=None, thresh=None):
    # TODO: refactor using make_ranking!!!
    if dists is None:
        dists = squareform(pdist(descriptors, distance))

    #fnames = [os.path.splitext(f)[0] for f in fnames]
    #os.makedirs(outdir, exist_ok=True)

    if not isinstance(thresh, dict):
        thresh = {i: thresh for i in range(scores.shape[1])}

    predclass = scores.argmax(axis=1)

    lens = Parallel(n_jobs=-1)(delayed(make_shrec17_output_thresh_loop)
                               (d, f, s, c, thresh, fnames, predclass, outdir)
                               for d, f, s, c in zip(dists, fnames, scores, predclass))

    print('avg # of elements returned {:2f} {:2f}'.format(np.mean(lens), np.std(lens)))


def make_shrec17_output_thresh_loop(d, f, s, c, thresh, fnames, predclass, outdir, max_retrieved=1000):
    t = thresh[c]

    fd = [(ff, dd)
          for dd, ff, cc in zip(d, fnames, predclass)
          # chose whether to include same class or not
          if (dd < t) or (cc == c)]
    # if (dd < t)]
    fi = [ff[0] for ff in fd]
    di = [ff[1] for ff in fd]

    ranking = []
    for i in np.argsort(di):
        if fi[i] not in ranking:
            ranking.append(fi[i].replace('.obj', ''))
    ranking = ranking[:max_retrieved]

    with open(os.path.join(outdir, f), 'w') as fout:
        [print(r, file=fout) for r in ranking]

    return len(ranking)


def search_thresholds(dists, labels):
    """ Search thresholds per class that maximizes F-score. """
    thresh = {i: [] for i in range(max(labels)+1)}
    dists /= dists.max()
    assert dists.min() >= 0
    assert dists.max() <= 1

    list_thresh = Parallel(n_jobs=-1)(delayed(search_thresholds_loop)(d, l, labels) for d, l in zip(dists, labels))

    for l, t in zip(labels, list_thresh):
        thresh[l].append(t)

    # mean thresh per class
    # these are 1-d, need to be more than that to be classified
    # d must be smaller than 1-this value
    thresh_mean = {i: 1-np.mean(t) for i, t in sorted(thresh.items())}

    return thresh_mean


def search_thresholds_loop(d, l, labels):
    p, r, t = precision_recall_curve(labels == l, 1-d)
    f = 2 * (p * r) / (p + r)

    return t[np.argmax(f)]


def eval_shrec17_output(outdir):
    basedir = Path(os.path.realpath(__file__)).parent
    evaldir = basedir / 'external_shrec17_evaluator'
    assert basedir.is_dir()
    assert evaldir.is_dir()
    assert os.path.isdir(outdir)
    evaldir = str(evaldir)
    # import ipdb; ipdb.set_trace()
    if outdir[-1] != '/':
        outdir += '/'
    # outdir_arg = os.path.join('../../', outdir)
    p = subprocess.Popen(['node', 'evaluate.js', outdir],
                         cwd=evaldir)
    p.wait()

    import pandas as pd
    data = pd.read_csv('{}/{}.summary.csv'
                       .format(evaldir, outdir.split('/')[-2]))

    return data


if __name__ == '__main__':
    with tf.Graph().as_default():
        retrieval()
    LOG_FOUT.close()
