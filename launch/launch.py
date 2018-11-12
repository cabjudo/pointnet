#!/bin/python3
import numpy as np
import os
import subprocess

# dataset = ['shrec17', 'modelnet40']
dataset = ['modelnet40']
model_choices = ['pointnet_notrans_add3x64','pointnet_notrans_add3x128','pointnet_notrans_add2x1024',"pointnet_notrans"]
representation = ["plane0", "darboux"]
train_test = ['z-z','so3-so3']

for data in dataset:
    for rep in representation:
        for arch in model_choices:
            for t in train_test:
                if 'z' in t and data in ['darboux']:
                    continue

                name = '{}-{}-{}-{}'.format(data[0], arch[:-9], rep[0], t)
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
                    '--representation={}'.format(rep),
                    '--model={}'.format(arch),
                    '--train_test={}'.format(t),
                    '--dataset={}'.format(data),
                    '--optimizer=adam',
                    '--batch_size=32',
                    '--num_point=1024',
                    '--max_epoch=150',
                    '--momentum=0.9',
                    '--decay_step=200000',
                    '--decay_rate=0.8'
                ]

            # Create yaml file
            kubectl_create_cmd = ['kubectl', 'create', '-f', '{}.yaml'.format(name)]
            # Run commands in shell
            subprocess.call(kcreator_cmd)
            # print('kcreator_cmd', kcreator_cmd)
            # subprocess.call(kubectl_create_cmd)
            # print('kubectl_create_cmd', kubectl_create_cmd)
            
# get_pods_cmd = ['kubectl', 'get', 'pods']
# Run commands in shell
# subprocess.call(get_pods_cmd)
