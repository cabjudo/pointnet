import numpy as np

from data_prep_chordiogram import plane2_aux
from data_prep_chordiogram import _prune_chords





def test_plane2_aux(n=10):
    p = np.random.rand(n,3)
    
    n_p = np.random.rand(n,3)
    n_p /= np.sum(n_p**2, axis=1).reshape(-1,1)

    q = np.random.rand(n,3)

    n_q = np.random.rand(n,3)
    n_1 /= np.sum(n_q**2, axis=1).reshape(-1,1)

    m = q - p

    test_chords = plane2_aux(m, n_p, n_q)
    chord = generate_chords(p, n_p, q, n_q, chord_type='DROST')

    assert np.allclose(chord, test_chord)


def generate_drost_chord(p, n_p, q, n_q):
    '''
    Generate drost descriptor
    d = q - p
    F(p, q) = (||d||, angle(n_p, d), angle(n_q, d), angle(n_p, n_q))

    Input
    p: (n, 3)
    n_p: (n, 3)

    q: (n, 3)
    n_q: (n, 3)
    '''
    d = q - p
    l = np.linalg.norm(d, axis=1)
    angle_np_d = np.arccos( np.sum(n_p * d, axis=1)/l )
    angle_nq_d = np.arccos( np.sum(n_q * d, axis=1)/l )
    angle_np_nq = np.arccos( np.sum(n_p * n_q, axis=1) )
    
    return (l, angle_np_d, angle_nq_d, angle_np_nq)



GenerateChordType = {'DROST': generate_drost_chord }





def generate_chords(p, n_p, q, n_q, chord_type='DROST'):
    return GenerateChordType['DROST'](p, n_p, q, n_q)


def generate_single_pointcloud_pair():
    '''
    Returns a chord description
    '''
    p = np.array([[0,1,0]])
    n_p = np.array([[0,1,0]])

    q = np.array([[1,0,0]])
    n_q = np.array([[1,0,0]])

    return p, n_p, q, n_q


def test_generate_chords():
    p, n_p, q, n_q = generate_single_pointcloud_pair()
    chord = generate_chords(p, n_p, q, n_q)

    assert np.allclose(chord[0], np.sqrt(2)), "chord[0]={}".format(chord[0])
    assert np.allclose(chord[1], 135.0 * np.pi/180.0), "chord[1]={}".format(chord[1])
    assert np.allclose(chord[2], 45.0 * np.pi/180.0), "chord[2]={}".format(chord[2])
    assert np.allclose(chord[3], 90.0 * np.pi/180.0), "chord[3]={}".format(chord[3])
    
    chord = generate_chords(q, n_q, p, n_p)

    assert np.allclose(chord[0], np.sqrt(2)), "chord[0]={}".format(chord[0])
    assert np.allclose(chord[1], 135.0 * np.pi/180.0), "chord[1]={}".format(chord[1])
    assert np.allclose(chord[2], 45.0 * np.pi/180.0), "chord[2]={}".format(chord[2])
    assert np.allclose(chord[3], 90.0 * np.pi/180.0), "chord[3]={}".format(chord[3])

    chord = generate_chords(p, n_q, q, n_p)

    assert np.allclose(chord[0], np.sqrt(2)), "chord[0]={}".format(chord[0])
    assert np.allclose(chord[1], 45.0 * np.pi/180.0), "chord[2]={}".format(chord[2])
    assert np.allclose(chord[2], 135.0 * np.pi/180.0), "chord[1]={}".format(chord[1])
    assert np.allclose(chord[3], 90.0 * np.pi/180.0), "chord[3]={}".format(chord[3])


def test_plane2_aux():
    pass



    
if __name__ == '__main__':
    # test_generate_chords()
    test_plane2_aux()
