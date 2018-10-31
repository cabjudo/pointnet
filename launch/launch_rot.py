#!/bin/python3
import numpy as np
import os
import subprocess

train_test_rot = ['z-z', 'z-so3', 'so3-so3']
model_choices = ["pointnet_cls", "pointnet_notrans"]
data_choices = ["original"]

for data in dataset_choices:
    for arch in model_choices:
        for train_test in train_test_rot:
            num = 0
            name = '{}-{}-rot'.format(data, arch[9:])
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
                '--optimizer=adam',
                '--batch_size=32',
                '--num_point=1024',
                '--max_epoch=250',
                '--momentum=0.9',
                '--decay_step=200000',
                '--decay_rate=0.8',
                '--train_test={}'.format(train_test)
            ]
            
            # create yaml file
            kubectl_create_cmd = ['kubectl', 'create', '-f', '{}.yaml'.format(name)]
            # Run commands in shell
            subprocess.call(kcreator_cmd)
            # print('kcreator_cmd', kcreator_cmd)
            subprocess.call(kubectl_create_cmd)
            # print('kubectl_create_cmd', kubectl_create_cmd)

get_pods_cmd = ['kubectl', 'get', 'pods']
# Run commands in shell
subprocess.call(get_pods_cmd)
