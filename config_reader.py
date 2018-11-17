import configparser
import argparse

config = configparser.ConfigParser()


def get_plane0(configfile):
    config.read(configfile)
    
    plane0 = dict(config.items('plane0'))
    plane0['num_chord_features'] = int(plane0['num_chord_features'])
    plane0['num_classes'] = int(plane0['num_classes'])

    return plane0


def get_plane1(configfile):
    config.read(configfile)
    
    plane1 = dict(config.items('plane1'))
    plane1['num_chord_features'] = int(plane1['num_chord_features'])
    plane1['num_classes'] = int(plane1['num_classes'])
    
    return plane1


def get_plane2(configfile):
    config.read(configfile)
    
    plane2 = dict(config.items('plane2'))
    plane2['num_chord_features'] = int(plane2['num_chord_features'])
    plane2['num_classes'] = int(plane2['num_classes'])
    
    return plane2


def get_original(configfile):
    config.read(configfile)
    
    original = dict(config.items('original'))
    original['num_chord_features'] = int(original['num_chord_features'])
    original['num_classes'] = int(original['num_classes'])
    
    return original


def get_darboux(configfile):
    config.read(configfile)
    
    darboux = dict(config.items('darboux'))
    darboux['num_chord_features'] = int(darboux['num_chord_features'])
    darboux['num_classes'] = int(darboux['num_classes'])
    
    return darboux


def get_darboux_aug(configfile):
    config.read(configfile)
    
    darboux_aug = dict(config.items('darboux_aug'))
    darboux_aug['num_chord_features'] = int(darboux_aug['num_chord_features'])
    darboux_aug['num_classes'] = int(darboux_aug['num_classes'])
    
    return darboux_aug


def get_darboux_expand_aug(configfile):
    config.read(configfile)
    
    darboux_expand_aug = dict(config.items('darboux_expand_aug'))
    darboux_expand_aug['num_chord_features'] = int(darboux_expand_aug['num_chord_features'])
    darboux_expand_aug['num_classes'] = int(darboux_expand_aug['num_classes'])
    
    return darboux_expand_aug


def get_darboux_sym(configfile):
    config.read(configfile)

    darboux_expand_sym = dict(config.items('darboux_sym'))
    darboux_expand_sym['num_chord_features'] = int(darboux_expand_sym['num_chord_features'])
    darboux_expand_sym['num_classes'] = int(darboux_expand_sym['num_classes'])

    return darboux_expand_sym


def get_darboux_expand(configfile):
    config.read(configfile)
    
    darboux_expand = dict(config.items('darboux_sym'))
    darboux_expand['num_chord_features'] = int(darboux_expand['num_chord_features'])
    darboux_expand['num_classes'] = int(darboux_expand['num_classes'])
    
    return darboux_expand


def get_representation(FLAGS):
    configfile = 'config/' + FLAGS.dataset + '.ini'
    if FLAGS.representation == 'plane0':
        rep = get_plane0(configfile)
    if FLAGS.representation == 'plane0':
        rep = get_plane2(configfile)
    if FLAGS.representation == 'darboux':
        rep = get_darboux(configfile)
    if FLAGS.representation == 'darboux_expand':
        rep = get_darboux_expand(configfile)
    if FLAGS.representation == 'darboux_aug':
        rep = get_darboux_aug(configfile)
    if FLAGS.representation == 'darboux_expand_aug':
        rep = get_darboux_expand_aug(configfile)
    if FLAGS.representation == 'darboux_sym':
        rep = get_darboux_sym(configfile)
    return rep
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse config file')
    parser.add_argument('configfile') 
    args = parser.parse_args()
    
    configfile = args.configfile
    # dataset
    
    plane0 = get_plane0(configfile)
    darboux = get_darboux(configfile)
    darboux_expand = get_darboux_expand(configfile)

    print(plane0)
    print(darboux)
    print(darboux_expand)
