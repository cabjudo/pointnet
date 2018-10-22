#!/bin/python3
import numpy as np
import os
import subprocess


model_choices = ["pointnet_notrans"]
data_choices = ["plane1", "plane2"]
lr = [ 5.3202e-04, 1.9118e-03 ] # best learning rate for plane1, plane2 respectively
indims = [256, 512, 1024, 2048]

for data, l in zip(data_choices, lr):
    for arch in model_choices:
       for num_point in indims:
           for num in range(1):
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
                    '--num_point={}'.format(num_point),
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
