import os
import sys
import argparse
import importlib

import tensorflow as tf

import config_reader


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

dataset_choices = ["modelnet40", 'modelnet40_aws',
                   "shrec17"]

rep_choices = ["plane0",
               "plane1",
               "plane2",
               "original",
               "darboux",
               "darboux_aug",
               "darboux_expand",
               "darboux_expand_aug",
               "darboux_sym"]

train_test = ["z-z",
              "z-so3",
              "so3-so3"]


def setup():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'models'))
    sys.path.append(os.path.join(base_dir, 'utils'))
    return base_dir


def load(FLAGS):
    # load model
    FLAGS.model_file = os.path.join(FLAGS.basedir, 'models', FLAGS.model+'.py')
    FLAGS.model = importlib.import_module(FLAGS.model) # import network module

    # create log dir
    if not os.path.exists(FLAGS.log_dir): os.mkdir(FLAGS.log_dir)
    os.system('cp %s %s' % (FLAGS.model_file, FLAGS.log_dir)) # bkp of model def
    os.system('cp train.py %s' % (FLAGS.log_dir)) # bkp of train procedure
    FLAGS.log_file = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w+')
    FLAGS.log_file.write(str(FLAGS)+'\n')

    return FLAGS

def model_path_parsing(FLAGS):
    # extract from model path
    if FLAGS.model_path is not None:
        param_string = FLAGS.model_path.split('/')[-2]
        # recover model
        for ind, m in enumerate(model_choices):
            m = m.replace('_','-')[-9:]
            if m in param_string:
                FLAGS.model = model_choices[ind]
                break
        # recover representation
        for ind, r in enumerate(rep_choices):
            if r in param_string:
                FLAGS.representation = rep_choices[ind]
                break
        # recover train_test
        if FLAGS.train_test is None:
            for ind, t in enumerate(train_test):
                if t in param_string:
                    FLAGS.train_test = train_test[ind]
                    break

    return FLAGS


def get_options():
    # get base directory
    basedir = setup()
    
    parser = argparse.ArgumentParser()
    # Model options
    parser.add_argument('--model', default='pointnet_cls', choices=model_choices, help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
    
    # Dataset options
    parser.add_argument('--representation', default='plane0', choices=rep_choices, help='Chordiogram representation [default: plane0]')
    parser.add_argument('--dataset', default='modelnet40', choices=dataset_choices,
                        help='Dataset [default: modelnet40]')
    
    # Dataset parameters
    parser.add_argument('--train_test', default=None, help='Train test setting: z-z]')
    parser.add_argument('--flip_train_test', default=False, help='Flips training and testing dataset')
    parser.add_argument('--augment', default=False, help='Augment the dataset')
    parser.add_argument('--drost', default=False, help='Augment the dataset')
    
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
    parser.add_argument('--triplet_loss', type=bool, default=True, help='Indicates whether to use a triplet loss at training. Only applies to shrec17 dataset')
    
    # Test options
    parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
    parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
    
    # Test params
    parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    
    # Logging options
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    
    # Logging params
    parser.add_argument('--save_freq', default=2, help='Save frequency in epochs')
    
    FLAGS = parser.parse_args()

    FLAGS = model_path_parsing(FLAGS)

    # Dataset load from config file
    representation = config_reader.get_representation(FLAGS)
    
    FLAGS.train_paths = representation['train']
    FLAGS.test_paths = representation['test']

    if FLAGS.augment:
        FLAGS.train_paths = representation['train_aug']
    if FLAGS.drost:
        FLAGS.train_paths = representation['train_drost']
        FLAGS.test_paths = representation['test_drost']

    if FLAGS.dataset == 'shrec17':
        FLAGS.retrieval_eval_path = representation['retrieval_eval']


    FLAGS.shape_names_path = os.path.join(os.path.dirname(os.path.dirname(representation['train'][0])), 'shape_names.txt')
    FLAGS.num_chord_features = representation['num_chord_features']
    FLAGS.num_classes = representation['num_classes']

    # add base directory to flags
    FLAGS.basedir = basedir

    # default flags
    FLAGS.max_num_point = 2048
    
    FLAGS.bn_init_decay = 0.5
    FLAGS.bn_decay_decay_rate = 0.5
    FLAGS.bn_decay_decay_step = float(FLAGS.decay_step)
    FLAGS.bn_decay_clip = 0.99

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    FLAGS.config = config

    FLAGS = load(FLAGS)

    return FLAGS


if __name__ == '__main__':
    FLAGS = get_options()
    print(FLAGS)
