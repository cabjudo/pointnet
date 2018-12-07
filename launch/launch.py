#!/bin/python3
import numpy as np
import os
import subprocess

<<<<<<< HEAD
lr = {}

samples = 5
# for learning rates, 10^-5 --> 10^-2 choose random numbers between 2 and 5
# np.random.rand returns values [0, 1)
lr_exponent = 2 + np.random.rand(samples) * 3
lr['point0'] = np.power(10, -lr_exponent)
lr_exponent = 2 + np.random.rand(samples) * 3
lr['darboux'] = np.power(10, -lr_exponent)

model_choices = [ "pointnet_notrans"]
data_choices = ["plane0", "darboux"]
test_train = ['z-z','z-so3','so3-so3']

for data in data_choices:
    for arch in model_choices:
        if 'z' in t and data in ['darboux']:
            continue

        for l in lr[data]:
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
                '--representation={}'.format(data),
                '--model={}'.format(arch),
                '--learning_rate={}'.format(l),
                '--train_test={}'.format(t)
                '--dataset=shrec17',
                '--optimizer=adam',
                '--batch_size=32',
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
=======
# dataset = ['shrec17', 'modelnet40']
dataset = ['modelnet40']
model_choices = ['pointnet_notrans_add3x64']
representation = ["plane2", "plane0"]
train_test = ['z-so3']

samples = 1
num_points = 1024

for data in dataset:
    for rep in representation:
        for arch in model_choices:
            for t in train_test:
                for s in range(samples):

                    lr_exponent = 1 + np.random.rand() * 4
                    lr = np.power(10, -lr_exponent)

                    if 'z' in t and data in ['darboux']:
                        continue

                    name = '{}-{}-{}-{}-drost-sym'.format(data[0], arch[17:], rep, t)
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
                        '--num_point={}'.format(num_points),
                        '--max_epoch=350',
                        '--momentum=0.9',
                        '--decay_step=200000',
                        '--decay_rate=0.8',
                        '--learning_rate=1e-3',
                        '--augment=True'
                    ]

                    # Create yaml file
                    kubectl_create_cmd = ['kubectl', 'create', '-f', '{}.yaml'.format(name)]
                    # Run commands in shell
                    subprocess.call(kcreator_cmd)
                    # print('kcreator_cmd', kcreator_cmd)
                    # subprocess.call(kubectl_create_cmd)
                    # print('kubectl_create_cmd', kubectl_create_cmd)
            
# get_pods_cmd = ['kubectl', 'get', 'pods']
>>>>>>> 2501d8aee86ac7a974cbde2e95df61fcbb2a30ed
# Run commands in shell
# subprocess.call(get_pods_cmd)
