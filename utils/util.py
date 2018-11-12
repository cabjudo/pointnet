import tensorflow as tf

import provider


def log_string(FLAGS, out_str):
    FLAGS.log_file.write(out_str+'\n')
    FLAGS.log_file.flush()
    print(out_str)


def get_learning_rate(FLAGS, batch):
    learning_rate = tf.train.exponential_decay(
                        FLAGS.learning_rate,  # Base learning rate.
                        batch * FLAGS.batch_size,  # Current index into the dataset.
                        FLAGS.decay_step,          # Decay step.
                        FLAGS.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(FLAGS, batch):
    bn_momentum = tf.train.exponential_decay(
                      FLAGS.bn_init_decay,
                      batch*FLAGS.batch_size,
                      FLAGS.bn_decay_decay_step,
                      FLAGS.bn_decay_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(FLAGS.bn_decay_clip, 1 - bn_momentum)
    return bn_decay


def perturb_data(FLAGS, data, mode='train'):
    if FLAGS.dataset in ["original"]:
        rotated_data = provider.rotate_point_cloud(data, mode, FLAGS.train_test)
        jittered_data = provider.jitter_point_cloud(rotated_data)
    elif FLAGS.dataset in ["plane0"]:
        rotated_data = provider.rotate_plane0_point_cloud(data, mode, FLAGS.train_test)
        jittered_data = provider.jitter_plane0(rotated_data)
    elif FLAGS.dataset in ["darboux_expand"]:
        rotated_data = provider.expand_darboux(data)
        jittered_data = provider.jitter_expand_darboux(rotated_data)
    else:
        rotated_data = data
        if FLAGS.dataset in ["plane1"]:
            jittered_data = provider.jitter_plane1(rotated_data)
        elif FLAGS.dataset in ["plane2"]:
            jittered_data = provider.jitter_plane2(rotated_data)
        else:
            jittered_data = provider.jitter_darboux(rotated_data)

    return rotated_data, jittered_data

