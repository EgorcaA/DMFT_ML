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

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from  tqdm import tqdm
import sys

import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

sys.path.append("./src")
import DataBase
import ml_model
import Sample
from util import *


beta = [20., 50.]
U = [2.0, 7.0]
n_entries = 60 #16 -> 20 min
filename ='./data/databaseV16.h5'

DB_get = DataBase.db_DMFT(beta, U, n_entries, filename)

DB_get.solve_db(n_workers=6)
DB_get.save_db()