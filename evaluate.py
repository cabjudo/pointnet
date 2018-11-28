import os
import sklearn
import sys
import argparse
import socket
import importlib
import time
import scipy.misc

import tensorflow as tf
import numpy as np

import options
import provider

from utils.util import perturb_data
from utils.util import log_string

FLAGS = options.get_options()

SHAPE_NAMES = [line.rstrip() for line in open(FLAGS.shape_names_path)]
TRAIN_FILES = FLAGS.train_paths
TEST_FILES = FLAGS.test_paths


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


def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(FLAGS.gpu)):
        pointclouds_pl, labels_pl = FLAGS.model.placeholder_inputs(FLAGS.batch_size, FLAGS.num_point, FLAGS.num_chord_features)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = FLAGS.model.get_model(pointclouds_pl, is_training_pl, input_dims=FLAGS.num_chord_features, num_classes=FLAGS.num_classes)
        loss = FLAGS.model.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)
        
    # Create a session
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # config.log_device_placement = True
    sess = tf.Session(config=FLAGS.config)

    # Restore variables from disk.
    saver.restore(sess, FLAGS.model_path)
    log_string(FLAGS, "Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes))
    total_seen_class = [0 for _ in range(FLAGS.num_classes)]
    total_correct_class = [0 for _ in range(FLAGS.num_classes)]
    fout = open(os.path.join(FLAGS.dump_dir, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string(FLAGS, '----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:FLAGS.num_point,:]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // FLAGS.batch_size
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx+1) * FLAGS.batch_size
            cur_batch_size = end_idx - start_idx
            
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, FLAGS.num_classes)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, FLAGS.num_classes)) # 0/1 for classes
            for vote_idx in range(num_votes):
                rotated_data, _ = perturb_data(FLAGS, current_data[start_idx:end_idx, :, :], 'test')

                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}

                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))

            # fills confustion matrix
            print(current_label[start_idx:end_idx].shape, np.argmax(pred_val, 1).shape)
            confusion_matrix += sklearn.metrics.confusion_matrix(current_label[start_idx:end_idx], np.argmax(pred_val, 1))

            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END
            
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                
                if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                    img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                           SHAPE_NAMES[pred_val[i-start_idx]])
                    img_filename = os.path.join(FLAGS.dump_dir, img_filename)
                    # output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                    # scipy.misc.imsave(img_filename, output_img)
                    error_cnt += 1

    # Saves confusion matrix
    np.save(os.path.join(FLAGS.model_path, '.confusion_matrix.npy'), confusion_matrix)
                
    log_string(FLAGS, 'eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string(FLAGS, 'eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string(FLAGS, 'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string(FLAGS, '%10s:\t%0.3f\t%d' % (name, class_accuracies[i], total_seen_class[i]))
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    FLAGS.log_file.close()
