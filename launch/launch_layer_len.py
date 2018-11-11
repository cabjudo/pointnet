#!/bin/python3
import numpy as np
import os
import subprocess

samples = 1
# for learning rates, 10^-5 --> 10^-2 choose random numbers between 2 and 5
# np.random.rand returns values [0, 1)
lr_exponent = 4 + np.random.rand(samples)
lr = np.power(10, -lr_exponent)

# model_choices = ['pointnet_notrans_add1024',
#                  'pointnet_notrans_add2x1024',
#                  'pointnet_notrans_add128',
#                  'pointnet_notrans_add2x128',
#                  'pointnet_notrans_add3x128',
#                  'pointnet_notrans_add64',
#                  'pointnet_notrans_add2x64',
#                  'pointnet_notrans_add3x64']

model_choices = ['pointnet_notrans_add2x1024']

data_choices = ["plane0", "darboux", "darboux_expand"]

for data in data_choices:
    for arch in model_choices:
        for num in range(1):
            for l in lr:
                name = '{}-{}-lr-{:.4e}'.format(data, arch[9:], l)
                name = name.replace('_', '-')
                name = name.replace('.', '-')
                name = name.replace('+', '')
                print('save-{}'.format(name))
                # Build command with hyperparameters specificed
                kcreator_cmd = [
                    'kcreator',
                    '-g', '1',
                    '--job-name', '{}'.format(name),
                    '-i', 'chaneyk/tensorflow:v1.10.0-py3',
                    '-w', '/NAS/home/projects/pointnet',
                    '-Ti', '-X',
                    '-nc', '1',
                    '-r', '10',
                    '--',
                    'python3', 'train.py',
                    '--log_dir=save/save-{}'.format(name),
                    '--dataset={}'.format(data),
                    '--model={}'.format(arch),
                    '--learning_rate={}'.format(l),
                    '--optimizer=adam',
                    '--batch_size=32',
                    '--augment=True',
                    '--num_point=1024',
                    '--max_epoch=250',
                    '--momentum=0.9',
                    '--decay_step=200000',
                    '--decay_rate=0.8'
                ]

                # Create yaml file
                kubectl_create_cmd = ['kubectl', 'create', '-f', '{}.yaml'.format(name)]
                # Run commands in shell
                subprocess.call(kcreator_cmd)
                # print('kcreator_cmd', kcreator_cmd)
                subprocess.call(kubectl_create_cmd)
                # print('kubectl_create_cmd', kubectl_create_cmd)

get_pods_cmd = ['kubectl', 'get', 'pods']
# Run commands in shell
subprocess.call(get_pods_cmd)
