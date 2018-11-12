import os
import sys
import argparse
import math
import h5py
import socket
import importlib

import numpy as np
import tensorflow as tf

import options
import provider

from utils import tf_util
from utils.util import log_string
from utils.util import get_learning_rate 
from utils.util import get_bn_decay
from utils.util import perturb_data


FLAGS = options.get_options()
TRAIN_FILES = provider.getDataFiles(FLAGS.train_path)
TEST_FILES = provider.getDataFiles(FLAGS.test_path)

# Flips the training and testing datasets
if FLAGS.flip_train_test:
    AUX_FLIP = FLAGS.train_path
    FLAGS.train_path = FLAGS.test_path
    FLAGS.test_path = AUX_FLIP



def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl, labels_pl = FLAGS.model.placeholder_inputs(FLAGS.batch_size, FLAGS.num_point, FLAGS.num_chord_features)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            epoch_counter = tf.Variable(0)
            eval_accuracy = tf.Variable(0)
            inc = tf.assign_add(epoch_counter, 1, name='increment')
            bn_decay = get_bn_decay(FLAGS, batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = FLAGS.model.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay,
                                               input_dims=FLAGS.num_chord_features)
            loss = FLAGS.model.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(FLAGS.batch_size)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(FLAGS, batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if FLAGS.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum)
            elif FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        sess = tf.Session(config=FLAGS.config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'test'))

        # Init variables
        # if a checkpoint exists, restore from the latest checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Init variables
            init = tf.global_variables_initializer()
            # To fix the bug introduced in TF 0.12.1 as in
            # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
            #sess.run(init)
            sess.run(init, {is_training_pl: True})

        #init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        #sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        assert sess.run(epoch_counter) < 250, 'Training is complete.'

        for epoch in range(sess.run(epoch_counter), FLAGS.max_epoch + 1):
            log_string(FLAGS, '**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            sess.run(inc)

            # Save the variables to disk.
            if epoch % FLAGS.save_freq == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"), global_step=batch)
                log_string(FLAGS, "Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """

    print("Using {} for training.".format(TRAIN_FILES))
    print("Using {} for testing.".format(TEST_FILES))

    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        log_string(FLAGS, '----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:FLAGS.num_point,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // FLAGS.batch_size

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx+1) * FLAGS.batch_size

            # Augment batched point clouds by rotation and jittering
            # rotation depends on dataset and train/test type
            rotated_data, jittered_data = perturb_data(FLAGS, current_data[start_idx:end_idx, :, :], 'train')

            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}

            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += FLAGS.batch_size
            loss_sum += loss_val

        log_string(FLAGS, 'mean loss: %f' % (loss_sum / float(num_batches)))
        log_string(FLAGS, 'accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class = [0 for _ in range(FLAGS.num_classes)]

    for fn in range(len(TEST_FILES)):
        log_string(FLAGS, '----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:FLAGS.num_point,:]
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // FLAGS.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx+1) * FLAGS.batch_size

            # Augment batched point clouds by rotation and jittering
            rotated_data, _ = perturb_data(FLAGS, current_data[start_idx:end_idx, :, :], 'test')
            
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}

            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)

            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += FLAGS.batch_size
            loss_sum += (loss_val*FLAGS.batch_size)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)


    log_string(FLAGS, 'eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string(FLAGS, 'eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string(FLAGS, 'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))



if __name__ == "__main__":
    train()
    FLAGS.log_file.close()
