import os
import sys
import argparse
import socket
import importlib
import time
import scipy.misc
import subprocess

import tensorflow as tf
import numpy as np

import options
import provider

from utils.util import perturb_data
from utils.util import log_string
from joblib import Parallel, delayed
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_curve, precision_score

FLAGS = options.get_options()

SHAPE_NAMES = [line.rstrip() for line in open(FLAGS.shape_names_path)]
TRAIN_FILES = provider.getDataFiles(FLAGS.train_path)
TEST_FILES = provider.getDataFiles(FLAGS.test_path)


# def log_string(out_str):
#     FLAGS.log_file.write(out_str+'\n')
#     FLAGS.log_file.flush()
#     print(out_str)


# def perturb_data(FLAGS, data, mode='train'):
#     if FLAGS.dataset in ["original"]:
#         rotated_data = provider.rotate_point_cloud(data, mode, FLAGS.train_test)
#         jittered_data = provider.jitter_point_cloud(rotated_data)
#     elif FLAGS.dataset in ["plane0"]:
#         rotated_data = provider.rotate_plane0_point_cloud(data, mode, FLAGS.train_test)
#         jittered_data = provider.jitter_plane0(rotated_data)
#     elif FLAGS.dataset in ["darboux_expand"]:
#         rotated_data = provider.expand_darboux(data)
#         jittered_data = provider.jitter_expand_darboux(rotated_data)
#     else:
#         rotated_data = data
#         if FLAGS.dataset in ["plane1"]:
#             jittered_data = provider.jitter_plane1(rotated_data)
#         elif FLAGS.dataset in ["plane2"]:
#             jittered_data = provider.jitter_plane2(rotated_data)
#         else:
#             jittered_data = provider.jitter_darboux(rotated_data)

#     return rotated_data, jittered_data



def retrieval():
    is_training = False
     
    with tf.device('/gpu:'+str(FLAGS.gpu)):
        pointclouds_pl, labels_pl = FLAGS.model.placeholder_inputs(FLAGS.batch_size, FLAGS.num_point,
                                                                   FLAGS.num_chord_features)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points, feature_map = FLAGS.model.get_model(pointclouds_pl, is_training_pl, input_dims=FLAGS.num_chord_features,
                                                 num_classes=FLAGS.num_classes, return_feature_map=True)
        loss = FLAGS.model.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=FLAGS.config)

    # Restore variables from disk.
    saver.restore(sess, FLAGS.model_path)
    log_string(FLAGS, "Model restored.")

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
    total_seen_class = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class = [0 for _ in range(FLAGS.num_classes)]
    fout = open(os.path.join(FLAGS.dump_dir, 'pred_label.txt'), 'w')
    all_descriptors = np.array([[]])
    scores = np.array([[]])
    labels = np.array([[]])
    fnames = np.array([])
    for fn in range(len(TEST_FILES)):
        log_string(FLAGS, '----'+str(fn)+'----')
        current_data, current_label, current_fnames = provider.loadDataFile(TEST_FILES[fn], return_fnames=True)
        current_data = current_data[:, 0:FLAGS.num_point, :]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // FLAGS.batch_size
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx+1) * FLAGS.batch_size
            cur_batch_size = end_idx - start_idx

            rotated_data, _ = perturb_data(FLAGS, current_data[start_idx:end_idx, :, :], 'test')

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

    fname = FLAGS.model_path + '_dists.npy'
    dists = compute_and_save_descriptors_dists(all_descriptors, fname)

    thresh = search_thresholds(dists, labels)

    out_dir = os.path.join(os.path.dirname(FLAGS.model_path), 'retrieval_out')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    make_shrec17_output_thresh(all_descriptors, scores, fnames, out_dir,
                               distance='cosine', dists=dists, thresh=thresh)

    res = eval_shrec17_output(os.path.split(FLAGS.log_dir)[0])
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

    #lens = []
    #for d, f, s, c in zip(dists, fnames, scores, predclass):
    #    lens.append(make_shrec17_output_thresh_loop(d, f, s, c, thresh, fnames, predclass, outdir))

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
            print(fi[i].decode('UTF-8'), type(fi[i].decode('UTF-8')))
            ranking.append(fi[i].decode('UTF-8').replace('.obj', ''))
    ranking = ranking[:max_retrieved]

    with open(os.path.join(outdir, f.decode('UTF-8')), 'w') as fout:
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
    print('Evaluating retrieval...')
    p = subprocess.Popen(['node', 'evaluate.js', outdir],
                         cwd=evaldir)
    print('Done.')
    p.wait()

    import pandas as pd
    data = pd.read_csv('{}/{}.summary.csv'
                       .format(evaldir, outdir.split('/')[-2]))

    return data


if __name__ == '__main__':
    with tf.Graph().as_default():
        retrieval()
    FLAGS.log_file.close()
