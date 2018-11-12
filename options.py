import os
import sys
import argparse

import tensorflow as tf


DatasetPath = {
    "plane0": {
        "train": '/NAS/data/diego/chords_dataset/plane0/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/plane0/test_files.txt',
        "num_chord_features": 7,
    },
    "plane1": {
        "train": '/NAS/data/diego/chords_dataset/plane1/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/plane1/test_files.txt',
        "num_chord_features": 3,
    },
    "plane2": {
        "train": '/NAS/data/diego/chords_dataset/plane2/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/plane2/test_files.txt',
        "num_chord_features": 4,
    },
    "original": {
        "train": '/NAS/data/christine/modelnet40_ply_hdf5_2048/train_files.txt',
        "test": '/NAS/data/christine/modelnet40_ply_hdf5_2048/test_files.txt',
        "num_chord_features": 3,
    },
    "darboux": {
        "train": '/NAS/data/diego/chords_dataset/darboux/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/darboux/test_files.txt',
        "num_chord_features": 4,
    },
    "darboux_expand": {
        "train": '/NAS/data/diego/chords_dataset/darboux/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/darboux/test_files.txt',
        "num_chord_features": 6,
    },
    "darboux_aug": {
        "train": '/NAS/data/diego/chords_dataset/darboux_aug/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/darboux_aug/test_files.txt',
        "num_chord_features": 5,
    },
    "darboux_expand_aug": {
        "train": '/NAS/data/diego/chords_dataset/darboux_aug/train_files.txt',
        "test": '/NAS/data/diego/chords_dataset/darboux_aug/test_files.txt',
        "num_chord_features": 6,
    },
}

model_choices = ["pointnet_cls",
                 "pointnet_cls_basic",
                 "pointnet_no3trans",
                 "pointnet_notrans",
                 'pointnet_notrans_add1024',
                 'pointnet_notrans_add2x1024',
                 'pointnet_notrans_add128',
                 'pointnet_notrans_add2x128',
                 'pointnet_notrans_add3x128',
                 'pointnet_notrans_add64',
                 'pointnet_notrans_add2x64',
                 'pointnet_notrans_add3x64']

dataset_choices = ["plane0",
                   "plane1",
                   "plane2",
                   "original",
                   "darboux",
                   "darboux_aug",
                   "darboux_expand",
                   "darboux_expand_aug"]

train_test = ["z-z",
              "z-so3",
              "so3-so3"]


def setup():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'models'))
    sys.path.append(os.path.join(base_dir, 'utils'))
    return base_dir


def get_options():
    # get base directory
    basedir = setup()
    
    parser = argparse.ArgumentParser()
    # Model options
    parser.add_argument('--model', default='pointnet_cls', choices=model_choices, help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
    
    # Dataset options
    parser.add_argument('--dataset', default='plane1', choices=dataset_choices, help='Dataset: chordiogram representation [default: plane11]')
    
    # Dataset parameters
    parser.add_argument('--train_test', default="z-z", help='Train test setting: z-z]')
    parser.add_argument('--flip_train_test', default=False, help='Flips training and testing dataset')
    parser.add_argument('--augment', default=False, help='Augment the dataset')
    
    # Training options
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    
    # Training parameters
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
    parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
    
    # Test options
    parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
    parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
    
    # Test params
    parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    
    # Logging options
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    
    # Logging params
    parser.add_argument('--save_freq', default=5, help='Save frequency in epochs')
    
    FLAGS = parser.parse_args()

    if FLAGS.augment:
        filepath_parts = DatasetPath[FLAGS.dataset]['train'].split('/')[:-1]
        filepath_parts += ['train_files_aug_5.txt']
        filepath = '/'.join(filepath_parts)
    else:    
        filepath_parts = DatasetPath[FLAGS.dataset]['train'].split('/')[:-1]
        filepath_parts += ['train_files_aug_1.txt']
        filepath = '/'.join(filepath_parts)

    DatasetPath[FLAGS.dataset]['train'] = filepath

    FLAGS.DatasetPath = DatasetPath

    # add base directory to flags
    FLAGS.basedir = basedir

    # default flags
    FLAGS.max_num_point = 2048
    FLAGS.num_classes = 40
    
    FLAGS.bn_init_decay = 0.5
    FLAGS.bn_decay_decay_rate = 0.5
    FLAGS.bn_decay_decay_step = float(FLAGS.decay_step)
    FLAGS.bn_decay_clip = 0.99

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    FLAGS.config = config

    




    return FLAGS


def load(FLAGS):
    # load model
    model = importlib.import_module(FLAGS.model) # import network module
    FLAGS.model_file = os.path.join(FLAGS.basedir, 'models', FLAGS.model+'.py')

    # create log dir
    if not os.path.exists(FLAGS.log_dir): os.mkdir(FLAGS.log_dir)
    os.system('cp %s %s' % (FLAGS.model_file, FLAGS.log_dir)) # bkp of model def
    os.system('cp train.py %s' % (FLAGS.log_dir)) # bkp of train procedure
    FLAGS.log_file = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w')
    FLAGS.log_file.write(str(FLAGS)+'\n')
