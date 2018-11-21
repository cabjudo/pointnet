import numpy as np



class PPFDescriptors(object):
    def __init__(self, mesh):
        self.mesh = mesh

        self.p = None
        self.n_p = None
        self.q = None
        self.n_q = None


    @static_method
    def _sample_chords(mesh, num_samples, sampling_method='random'):
        p, n_p, q, n_q = _oversample_chords(num_samples, sampling_method='random')
        p, n_p, q, n_q = _prune_chords(p, n_p, q, n_q)
        
        self.p = p[:num_samples]
        self.n_p = n_p[:num_samples]
        self.q = q[:num_samples]
        self.n_q = n_q[:num_samples]


    @static_method    
    def _oversample_chords(mesh, num_samples, sampling_method='random'):
        # over sample faces since we reject some
        num_sample_faces = 2*num_samples
        
        # associate a probility with each face depending on the area
        if sampling_method == 'area_weighted':
            prob = mesh.area_faces / mesh.area
        else:
            prob = np.ones(mesh.faces.shape[0]) / mesh.faces.shape[0]
            
        # randomly select pairs of faces for chord generation
        # random choice depends on the probability
        chord_pairs = np.random.choice(mesh.faces.shape[0], size=(num_sample_faces, 2), p=prob)
        
        # get face p and q
        face_p = mesh.faces[chord_pairs[:, 0], :]
        face_q = mesh.faces[chord_pairs[:, 1], :]
        # get normal p and q
        n_p = mesh.face_normals[chord_pairs[:, 0], :]
        n_q = mesh.face_normals[chord_pairs[:, 1], :]
        # get point p and q; random convex combination of face vertices
        p = _compute_point_in_triangle(mesh, num_sample_faces, face_p)
        q = _compute_point_in_triangle(mesh, num_sample_faces, face_q)

        return p, n_p, q, n_q


    @static_method
    def _prune_chords(p, n_p, q, n_q):
        '''
        Prunes chords according to Hinterstoisser et al. Going Further with Point Pair Features
        '''
        s = q - p # segment
        
        # Filter chords
        # length filter
        l = np.linalg.norm(s, axis=1)
        length_cond = l > 0.1
        # angle btwn normals
        angle_cond = np.arccos(np.sum(n_p * n_q, axis=1)) > 30.0 * np.pi/180.0
        
        idx = np.where(np.logical_or(length_cond, angle_cond))[0]
        
        return p[idx,:], n_p[idx,:], q[idx,:], n_q[idx,:]

    
    def _compute_point_in_triangle(mesh, num_sample_faces, face_indexes):
        # random three vector normalized, values give convex combination
        aux = np.random.rand(num_sample_faces, 3)
        aux /= np.dot(aux.sum(axis=1)[:, None], np.ones((1, 3)))

        # 
        lambdas = np.stack((aux, aux, aux), axis=1)
        points1_left = mesh.vertices[face_indexes[:, 0], :]
        points2_left = mesh.vertices[face_indexes[:, 1], :]
        points3_left = mesh.vertices[face_indexes[:, 2], :]
        k = np.stack((points1_left, points2_left, points3_left), axis=2) * lambdas
        points = np.sum(k, axis=2)
        return points




class Drost():
    pass
