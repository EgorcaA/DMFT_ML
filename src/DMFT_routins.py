################################################################################
#
# TPRF: Two-Particle Response Function (TPRF) Toolbox for TRIQS
#
# Copyright (C) 2019 by The Simons Foundation
# Author: H. U.R. Strand
#
# TPRF is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TPRF is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TPRF. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import copy
import glob
import numpy as np

from triqs.operators import n
# from pytriqs.archive import HDFArchive
from triqs.gf import Gf, MeshImFreq, Fourier, LegendreToMatsubara, MatsubaraToLegendre, BlockGf, inverse, Idx
from triqs.gf.tools import fit_legendre
from h5 import *
from triqs.gf import *
import time
import triqs_cthyb

from triqs_tprf.lattice import lattice_dyson_g_w
from triqs_tprf.ParameterCollection import ParameterCollection
from triqs_tprf.ParameterCollection import ParameterCollections
from triqs_tprf.utilities import  BlockGf_data

from triqs_tprf.tight_binding import create_square_lattice

import torch
import torch.nn as nn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt

def setup_dmft_calculation(p, BS):

    p = copy.deepcopy(p)
    p.iter = 0

    # -- Local Hubbard interaction
    
    p.solve.h_int = p.U*n('up', 0)*n('dn', 0)

    # -- 2D square lattice w. nearest neighbour hopping t
    square_lattice = create_square_lattice(norb=1, t=1.0)
    nk = 10
    dim = 2
    kmesh = square_lattice.get_kmesh(n_k=[nk] * dim + [1] * (3 - dim))
    p.e_k = square_lattice.fourier(kmesh)
    p.e_k.data[:] = BS
    # print(p.e_k)
    # p.e_k = H.on_mesh_brillouin_zone(n_k = (p.n_k, p.n_k, 1))

    # -- Initial zero guess for the self-energy    

    # p.sigma_l = GfLegendre(indices = [1], beta = p.init.beta, n_points = p.init.n_l)
    p.sigma_w = Gf(mesh=MeshImFreq(p.init.beta, 'Fermion', p.init.n_iw), target_shape=[1, 1])
    p.sigma_w.zero()

    p.g0_l = GfLegendre(indices = [1], beta = p.init.beta, n_points = p.init.n_l)
    p.g0_tau = Gf(mesh=MeshImTime(beta=p.init.beta, statistic='Fermion', n_tau=p.init.n_tau), target_shape=[1, 1])

    p.g_l = GfLegendre(indices = [1], beta = p.init.beta, n_points = p.init.n_l)
    p.g_tau = Gf(mesh=MeshImTime(beta=p.init.beta, statistic='Fermion', n_tau=p.init.n_tau), target_shape=[1, 1])

    return p

def solve_self_consistent_dmft(p):

    ps = []
    for dmft_iter in range(p.n_iter):
        print('--> DMFT Iteration: {:d}'.format(p.iter))
        p = dmft_self_consistent_step(p)
        ps.append(p)
        print('--> DMFT Convergence: dG_l = {:f}'.format(p.dG_l))
        if p.dG_l < p.G_l_tol: break

    if dmft_iter >= p.n_iter - 1: print('--> Warning: DMFT Not converged!')
    else: print('--> DMFT Converged: dG_l = {:f}'.format(p.dG_l))
    return ps

def solve_self_consistent_dmftML(p, MLmodel):

    ps = []
    for dmft_iter in range(p.n_iter):
        print('--> DMFT Iteration: {:d}'.format(p.iter))
        p = MLdmft_self_consistent_step(p, MLmodel)
        ps.append(p)
        print('--> DMFT Convergence: dG_l = {:f}'.format(p.dG_l))
        if p.dG_l < p.G_l_tol: break

    if dmft_iter >= p.n_iter - 1: print('--> Warning: DMFT Not converged!')
    else: print('--> DMFT Converged: dG_l = {:f}'.format(p.dG_l))
    return ps



def dmft_self_consistent_step(p):

    '''
    p: 
    sigma_w
    g0_l
    g0_tau
    g_l
    g_tau
    '''
    p = copy.deepcopy(p)
    p.iter += 1

    g_w = lattice_dyson_g_w(p.mu, p.e_k, p.sigma_w) # Mats
    g0_w = g_w.copy()
    g0_w << inverse(inverse(g_w) + p.sigma_w)
    
    # p.g0_l << MatsubaraToLegendre(g0_w)
    p.g0_tau << Fourier(g0_w)
    p.g0_l = fit_legendre(p.g0_tau, p.init.n_l)
    # oplot(g0_w)

    cthyb = triqs_cthyb.Solver(**p.init.dict())
    cthyb.G0_iw['up'] << g0_w
    cthyb.G0_iw['dn'] << g0_w

    
    start = time.time()
    cthyb.solve(**p.solve.dict())
    end = time.time()
    p.auto_corr_time =  cthyb.auto_corr_time
    if cthyb.auto_corr_time > 1.5:
        p.solve.length_cycle = int(p.solve.length_cycle*2.0)  
    if cthyb.auto_corr_time < 0.7:  
        p.solve.length_cycle = int( p.solve.length_cycle/1.5)
    p.time2solve = end - start
    p.averageorder = cthyb.average_order
        
    # print(p.iter)
    Gl = (cthyb.G_l['up'] + cthyb.G_l['dn'])*0.5
    p.dG_l = np.max(np.abs( (Gl- p.g_l).data.flatten() )) if hasattr(p,'g_l')  else float('nan')
    
    G_tau = cthyb.G_tau.copy()
    G0_w, G_w, Sigma_w = cthyb.G0_iw.copy(), cthyb.G_iw.copy(), cthyb.G0_iw.copy()

    #------------- optional to kill the noise
    G_tau << LegendreToMatsubara(cthyb.G_l)
    G_w << Fourier(G_tau)
    #------------- optional


    Sigma_w << inverse(G0_w) - inverse(G_w)

    # -- set lattice from impurity
    p.sigma_w << (Sigma_w['up'] + Sigma_w['dn'])*0.5
    # p.sigma_l << MatsubaraToLegendre(p.sigma_w)
    p.g_tau << (G_tau['up'] + G_tau['dn'])*0.5
    p.g_l << Gl

    return p



def MLdmft_self_consistent_step(p, MLmodel):

    '''
    sigma_w
    g0_l
    g0_tau
    g_l
    g_tau
    
    '''
    p = copy.deepcopy(p)
    p.iter += 1

    #SETUP--BEG-------
    Gtmp_iw = Gf(mesh=MeshImFreq(beta=p.init.beta, S='Fermion', n_iw=p.init.n_iw), target_shape=[1,1])
    G_w     = Gtmp_iw.copy()
    
    G_l_predicted = Gf(mesh=MeshLegendre(p.init.beta, 'Fermion', 30), target_shape=[1,1])
    #SETUP--END-------


    # print(p.sigma_w.data.flatten()[:3])
    g_w = lattice_dyson_g_w(p.mu, p.e_k, p.sigma_w) # Mats
    g0_w = g_w.copy()
    g0_w << inverse(inverse(g_w) + p.sigma_w)
    # print(g0_w.data.flatten()[:3])

    
    p.g0_l << MatsubaraToLegendre(g0_w)

    features = np.hstack([p.U, p.init.beta, p.g0_l.data.flatten()])
    leg = MLmodel(torch.Tensor(features)).cpu().detach().numpy()
    G_l_predicted.data[:] = leg[:, np.newaxis, np.newaxis]
    # print(G_l_predicted.data.flatten()[:3])

    p.dG_l = np.max(np.abs((G_l_predicted-p.g_l).data.flatten())) if hasattr(p,'g_l') else float('nan')
    p.g_l = G_l_predicted


    p.g_tau << LegendreToMatsubara(p.g_l)
    G_w << Fourier(p.g_tau)
    p.sigma_w << inverse(g0_w) - inverse(G_w)

    
    return p



# def MLdmft_self_consistent_step(p, MLmodel):

#     '''
#     sigma_w
#     g0_l
#     g0_tau
#     g_l
#     g_tau
    
#     '''
#     p = copy.deepcopy(p)
#     p.iter += 1

#     #SETUP--BEG-------
#     Gtmp_tau = Gf(mesh=MeshImTime(beta=p.init.beta, statistic='Fermion', n_tau=p.init.n_tau), target_shape=[1,1])
#     Gtmp_iw = Gf(mesh=MeshImFreq(beta=p.init.beta, S='Fermion', n_iw=p.init.n_iw), target_shape=[1,1])
    
#     p.g_tau, = Gtmp_tau.copy()
#     G_w, p.sigma_w     = Gtmp_iw.copy(), Gtmp_iw.copy()
    
#     G0_l = GfLegendre(indices = [1], beta = p.init.beta, n_points = p.init.n_l)

#     G_l_predicted = Gf(mesh=MeshLegendre(p.init.beta, 'Fermion', 30), target_shape=[1,1])
#     #SETUP--END-------


#     g_w = lattice_dyson_g_w(p.mu, p.e_k, p.sigma_w) # Mats
#     g0_w = g_w.copy()
#     g0_w << inverse(inverse(g_w) + p.sigma_w)

    
#     G0_l << MatsubaraToLegendre(p.g0_w)

#     features = np.hstack([p.U, p.init.beta, G0_l.data.flatten()])
#     leg = MLmodel(torch.Tensor(features)).cpu().detach().numpy()
#     G_l_predicted.data[:] = leg[:, np.newaxis, np.newaxis]


#     p.dG_l = np.max(np.abs((G_l_predicted-p.G_l).data.flatten())) if hasattr(p,'G_l') else float('nan')
#     p.G_l = G_l_predicted


#     p.G_tau << LegendreToMatsubara(p.G_l)
#     p.G_w << Fourier(p.G_tau)
#     p.Sigma_w << inverse(p.g0_w) - inverse(p.G_w)

#     # -- set lattice from impurity
#     p.sigma_w << p.Sigma_w
#     p.g_tau << p.G_tau

#     # -- local observables
#     p.rho = p.g_w.density()
#     # print(p.rho)
#     M_old = p.M if hasattr(p, 'M') else float('nan')
#     p.M = p.rho[0,0]
#     p.dM = np.abs(p.M - M_old)
    
#     return p



# def dmft_self_consistent_step(p):

#     p = copy.deepcopy(p)
#     p.iter += 1

#     p.g_w = lattice_dyson_g_w(p.mu, p.e_k, p.sigma_w)
#     p.g0_w = p.g_w.copy()
#     p.g0_w << inverse(inverse(p.g_w) + p.sigma_w)

#     cthyb = triqs_cthyb.Solver(**p.init.dict())

#     # print(p.g0_w)
#     # print(cthyb.G0_iw['up'])
#     # -- set impurity from lattice
#     cthyb.G0_iw['up'] << p.g0_w
#     cthyb.G0_iw['dn'] << p.g0_w

#     cthyb.solve(**p.solve.dict())

#     p.dG_l = np.max(np.abs(BlockGf_data(cthyb.G_l-p.G_l))) if hasattr(p,'G_l') else float('nan')
#     p.G_l = cthyb.G_l

#     p.G_tau, p.G_tau_raw = cthyb.G_tau.copy(), cthyb.G_tau.copy()
#     p.G0_w, p.G_w, p.Sigma_w = cthyb.G0_iw.copy(), cthyb.G0_iw.copy(), cthyb.G0_iw.copy()

#     p.G_tau << LegendreToMatsubara(p.G_l)
#     p.G_w << Fourier(p.G_tau)
#     p.Sigma_w << inverse(p.G0_w) - inverse(p.G_w)

#     # -- set lattice from impurity
#     p.sigma_w << (p.Sigma_w['up'] + p.Sigma_w['dn'])*0.5

#     # -- local observables
#     p.rho = p.g_w.density()
#     print(p.rho)
#     M_old = p.M if hasattr(p, 'M') else float('nan')
#     p.M = p.rho[0,0]
#     p.dM = np.abs(p.M - M_old)
    
#     return p
