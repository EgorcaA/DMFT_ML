from triqs.operators import *
from triqs_cthyb import Solver

# from triqs.gf import  MeshImTime, MeshReFreq, iOmega_n, inverse, GfLegendre, MeshImFreq, Gf, GfImFreq, GfImTime, Fourier
from triqs.gf import *
from triqs.gf.descriptors import MatsubaraToLegendre
# from triqs.plot.mpl_interface import plt

from h5 import *
from functools import partial
from multiprocessing import Pool, Manager
import numpy as np

import matplotlib.pyplot as plt
from  tqdm import tqdm
import sys
import time

from util import * 
from triqs_maxent import *
from triqs_tprf.lattice import lattice_dyson_g_w
from triqs_tprf.ParameterCollection import ParameterCollection
from triqs_tprf.ParameterCollection import ParameterCollections
from triqs_tprf.utilities import BlockGf_data

from DMFT_routins import *



class sample():
    def __init__(self, 
                 beta=5.0, 
                 U=2.0, 
                 BS=None,
                 pc=None
                 ):

        self.beta = beta
        self.U = U
        self.BS = BS
        
        assert BS is not None, "BS not defined!"

        if pc is not None:
            self.ps_exact = pc
        else:
            self.ps_exact = 0
            self.setup_DMFT()

        
    @classmethod
    def fromdict(cls, datadict):
        "Initialize sample from a dict's items"
        return cls(**datadict)

    def save2dict(self):
        dict2save = {}
        dict2save['beta'] = self.beta 
        dict2save['U'] = self.U
        dict2save['BS'] = self.BS 
        dict2save['pc'] = ParameterCollections(self.ps_exact)
        return dict2save



    def setup_DMFT(self):
        p = ParameterCollection(
            # t = 1.,
            # B = 0.,
            U = self.U,
            mu = 0.,
            n_k = 10,
            n_iter = 7,
            G_l_tol = 5e-3
            )

        p.solve = ParameterCollection(
            length_cycle = 15,
            n_warmup_cycles = 3000,
            n_cycles = int(5e6),
            move_double = False,
            measure_G_l = True,
            verbosity= 2
            )

        p.init = ParameterCollection(
            beta = self.beta,
            n_l = 50,
            n_iw = 400,
            n_tau = 4001,
            gf_struct = [('up', 1), ('dn', 1)]
            )

        self.p0 = setup_dmft_calculation(p, self.BS[:, np.newaxis,  np.newaxis])


    def solve_SC_dmft(self):
        p = copy.deepcopy(self.p0)

        ps = []
        for dmft_iter in range(p.n_iter):
            print('--> DMFT Iteration: {:d}'.format(p.iter))
            p = dmft_self_consistent_step(p)
            ps.append(p)
            print('--> DMFT Convergence: dG_l = {:f}'.format(p.dG_l))
            if p.dG_l < p.G_l_tol: break

        if dmft_iter >= p.n_iter - 1: print('--> Warning: DMFT Not converged!')
        else: print('--> DMFT Converged: dG_l = {:f}'.format(p.dG_l))
        self.ps_exact = ps
        # return ps
    