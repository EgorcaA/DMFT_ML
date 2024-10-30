from triqs.gf import *
from triqs.operators import *
from triqs.gf import Fourier
from h5 import *
import triqs.utility.mpi as mpi

from triqs.operators import *
from triqs_cthyb import Solver

from triqs.plot.mpl_interface import oplot,plt
from triqs.gf import MeshImTime
from triqs.gf import GfImFreq, GfImTime, make_gf_from_fourier

from triqs.gf import  MeshImTime, MeshReFreq, Gf
from triqs.plot.mpl_interface import oplot, plt

from multiprocessing import Pool
from itertools import repeat
from functools import partial

import csv
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# plt.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import warnings
warnings.filterwarnings('ignore')

# from pymatgen.electronic_structure.plotter import BSDOSPlotter, BSPlotter, DosPlotter
# from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

import pickle 
from  tqdm import tqdm

import sys
import torch
import torch.nn as nn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


import copy
import glob
import numpy as np

from triqs.gf import Gf, MeshImFreq, Fourier, LegendreToMatsubara, BlockGf, inverse, Idx

import triqs_cthyb

from triqs_tprf.lattice import lattice_dyson_g_w
from triqs_tprf.ParameterCollection import ParameterCollection
from triqs_tprf.ParameterCollection import ParameterCollections
from triqs_tprf.utilities import BlockGf_data
sys.path.append("./src")

# import DataBase
# import ml_model
# import Sample 
from band_generation import Band_generator
from util import * 
from ml_model import MLSolver   
import Sample
from DMFT_routins import *

bg = Band_generator()
bg.load_checkpoint('./models/AUTOmodel_v4')
BS = bg.getBS().flatten()      

model_path='./models/model_v3'
MLmodel = MLSolver().to(device, torch.float32)
MLmodel.load_checkpoint(checkpoint_path=model_path)

beta= 10.
s = Sample.sample(beta=beta,U=2.0, BS=BS)
s.setup_DMFT()
s.solve_SC_dmft()

