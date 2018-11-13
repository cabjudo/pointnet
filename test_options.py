import os
import sys
import argparse
import importlib

import tensorflow as tf

import options

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

dataset_choices = ["modelnet40",
                   "shrec17"]

rep_choices = ["plane0",
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

class test_flag(object):
    def __init__(self):
        self.model = None
        self.representation = None
        self.train_test = None
        self.model_path = None


def construct_model_path_string(FLAGS):
    model_path = FLAGS.model.replace('_','-') + FLAGS.representation + FLAGS.train_test + '/something'
    f = test_flag()
    f.model_path = model_path
    print(f.model_path)
    return f

    
def test_model_path_parsing(FLAGS):
    f = construct_model_path_string(FLAGS)
    f = options.model_path_parsing(f)

    assert f.model == FLAGS.model, 'parsing model param failed'
    assert f.train_test == FLAGS.train_test, 'parsing train_test param failed'
    assert f.representation == FLAGS.representation, 'parsing representation param failed'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pointnet_cls', choices=model_choices, help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
    parser.add_argument('--representation', default='plane0', choices=rep_choices, help='Chordiogram representation [default: plane0]')
    parser.add_argument('--train_test', default="z-z", help='Train test setting: z-z]')
    parser.add_argument('--model_path', default=None, help='Train test setting: z-z]')
    
    FLAGS = parser.parse_args()

    test_model_path_parsing(FLAGS)

