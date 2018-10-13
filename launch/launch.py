#!/bin/python3
import numpy as np
import os
import subprocess



samples = 20
# for learning rates, 10^-5 --> 10^-2 choose random numbers between 2 and 5
# np.random.rand returns values [0, 1)
lr_exponent = 2 + np.random.rand(samples)*3
lr = np.power(10, -lr_exponent)

model_choices = ["pointnet_cls", "pointnet_no3trans", "pointnet_notrans"]
data_choices = ["uniform", "area_wieghted"]

for data in data_choices:
  for arch in model_choices:
    for num in range(5):
      for l in lr:
        name='{}-{}-lr-{:.4e}'.format(data, arch[9:], l)
        name = name.replace('_','-')
        name = name.replace('.','-')
        name = name.replace('+','')
        print('save-{}'.format(name))
        # Build command with hyperparameters specificed
        kcreator_cmd = [
          'kcreator',
          '-g','1' ,
          '--job-name', '{}'.format(name),
          '-i', 'chaneyk/tensorflow:a4f7556-py3',
          '-w', '/NAS/home/pointnet',
          '-Ti','-X',
          '-nc','1',
          '--',
          'python3', 'train.py',
          '--log_dir=save/save-{}'.format(name),
          '--dataset={}'.format(data),
          '--model={}'.format(arch),
          '--learning_rate={}'.format(l),
          '--optimizer_type=adam',
          '--batch_size=64',
        ]
        # Create yaml file
        kubectl_create_cmd = [ 'kubectl', 'create', '-f', '{}.yaml'.format(name) ]
        # Run commands in shell
        subprocess.call(kcreator_cmd)
        # print('kcreator_cmd', kcreator_cmd)
        subprocess.call(kubectl_create_cmd)
        # print('kubectl_create_cmd', kubectl_create_cmd)

get_pods_cmd = [ 'kubectl', 'get', 'pods' ]
# Run commands in shell
subprocess.call(get_pods_cmd)


