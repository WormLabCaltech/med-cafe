"""
"""
import numpy as np


class Model:
    """
    A phenotypic class model. Given the sizes of a set of 8 classes and noise
    parameters, builds an observation matrix and simulates noise from
    false-positive and false-negative flows. It can also test an idealized
    model after noising with an observed model (must be provided) and return a
    signal to noise estimate for each class.

    properties:
    L
    fp
    fn
    fps
    fns
    N_T

    """
    def __init__(self, N_100, N_010, N_001, N_T, M100, M010, M001, M110, M101,
                 M011, M111, fp, fn):
        """
        Initialization function for a `model` object.

        params:
        N_100, N_010, N_001
        M_ijk
        fp
        fn
        """
        self.N = np.array([N_100, N_010, N_001])
        self.N_T = N_T
        self.M100, self.M010, self.M001 = M100, M010, M001
        self.M110, self.M101, self.M011, self.M111 = M110, M101, M011, M111
        # index matrix, standard for all 3-way comparisons
        L = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 1, 0],
             [1, 0, 1],
             [0, 1, 1],
             [1, 1, 1],
             [0, 0, 0]]
        L = np.matrix(L)
        self.L = L

        # adjacency matrix
        self.A = [[0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 0],
                  [1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 1],
                  [0, 1, 1, 0, 0, 0, 1]
                  ]

        # parameters:
        self.fp = fp
        self.fn = fn

        # initialize the M classes:
        self.make_M()

        self.make_fns()
        self.make_fps()

    def make_M(self):
        """
        Inputs Mijk entries into 3x3 array, and inflates them for false
        negatives.
        """
        # total DE genes counted:
        DE = self.M100 + self.M010 + self.M001 + self.M110 + self.M101 +\
            self.M011 + self.M111

        # make matrix and input coefficients:
        M_mat = np.zeros(shape=(2, 2, 2))

        M_mat[1, 0, 0] = self.M100
        M_mat[0, 1, 0] = self.M010
        M_mat[0, 0, 1] = self.M001
        M_mat[1, 1, 0] = self.M110
        M_mat[0, 1, 1] = self.M011
        M_mat[1, 0, 1] = self.M101
        M_mat[1, 1, 1] = self.M111
        M_mat[0, 0, 0] = self.N_T - DE

        self.M = M_mat

    def find_labels(self, t):
        """Given a genotype t, find all the labels that include that
        genotype"""
        return self.L[np.where(self.L[:, t] == 1)[0]]

    def find_sub_M(self, t):
        """Find the submatrix of M that only contains entries that include
        genotype t"""
        return self.M[np.where(self.L[:, t] == 1)[0]]

    def make_fps(self):
        self.fps = np.zeros(shape=(2, 2, 2))

        fps000 = -self.M[0, 0, 0]*self.N.sum()
        fps100 = self.M[0, 0, 0]*self.N[0] - self.M100*(self.N[1] + self.N[2])
        fps010 = self.M[0, 0, 0]*self.N[1] - self.M010*(self.N[0] + self.N[2])
        fps001 = self.M[0, 0, 0]*self.N[2] - self.M001*(self.N[0] + self.N[1])
        fps110 = self.M100*self.N[1] + self.M010*self.N[0] -\
            self.M110*self.N[2]
        fps101 = self.M100*self.N[2] + self.M001*self.N[0] -\
            self.M101*self.N[1]
        fps011 = self.M010*self.N[2] + self.M001*self.N[1] -\
            self.M011*self.N[0]
        fps111 = self.M110*self.N[2] + self.M101*self.N[1] +\
            self.M011*self.N[0]

        # make matrix and input coefficients:
        self.fps = np.zeros(shape=(2, 2, 2))

        self.fps[1, 0, 0] = fps100
        self.fps[0, 1, 0] = fps010
        self.fps[0, 0, 1] = fps001
        self.fps[1, 1, 0] = fps110
        self.fps[0, 1, 1] = fps011
        self.fps[1, 0, 1] = fps101
        self.fps[1, 1, 1] = fps111
        self.fps[0, 0, 0] = fps000

        self.fps = self.fps*self.fp/self.N_T

    def make_fns(self):
        self.fns = np.zeros(shape=(2, 2, 2))

        fns000 = self.M100 + self.M010 + self.M001
        fns100 = self.M110 + self.M101 - self.M100
        fns010 = self.M110 + self.M011 - self.M010
        fns001 = self.M101 + self.M011 - self.M001
        fns110 = self.M111 - self.M110*2
        fns101 = self.M111 - self.M101*2
        fns011 = self.M111 - self.M011*2
        fns111 = -self.M111*3

        # make matrix and input coefficients:
        self.fns = np.zeros(shape=(2, 2, 2))

        self.fns[1, 0, 0] = fns100
        self.fns[0, 1, 0] = fns010
        self.fns[0, 0, 1] = fns001
        self.fns[1, 1, 0] = fns110
        self.fns[0, 1, 1] = fns011
        self.fns[1, 0, 1] = fns101
        self.fns[1, 1, 1] = fns111
        self.fns[0, 0, 0] = fns000

        self.fns = self.fns*self.fn

    def signal_threshold(self, alpha):
        """Sets the signal/noise threshold for testing."""
        self.snr = alpha

    def test_classes(self, M_obs):
        """Tests classes against observations and accepts classes based on an
        alpha value."""
        accepted = np.array([0]*7)
        sn = np.array([0.]*7)
        SN = M_obs/(self.fps + self.fns + .01)
        SN[SN > 10**6] = 10**6  # set a roof

        # accept high signal classes
        for l in self.L:
            if l.sum() == 0:
                continue
            index = (l[0, 0], l[0, 1], l[0, 2])
            signal = SN[index]
            if (signal > self.snr) | (signal < 0):
                # find which entry this is in
                # the column is useless
                row, col = np.where(np.all(self.L == l, axis=1))
                accepted[row] = M_obs[index]
                sn[row] = signal
        self.accepted = accepted
        self.signal = sn
