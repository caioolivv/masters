# For NumCosmo
import os
import sys
import gi

gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import GObject
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

from scipy.stats import chi2

import math
# The corner package is needed to view the results of the MCMC analysis
import corner


os.environ['CLMM_MODELING_BACKEND'] = 'nc'

__name__ = "NcContext"

Ncm.cfg_init ()
Ncm.cfg_set_log_handler (lambda msg: sys.stdout.write (msg) and sys.stdout.flush ())

try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from clmm.support.sampler import fitters

import clmm.dataops as da
import clmm.galaxycluster as gc
import clmm.theory as theory
from clmm import Cosmology
from clmm.support import mock_data as mock
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

np.random.seed(11)

mock_cosmo = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)

cosmo = mock_cosmo
cluster_m = 1.e15 # M200,m [Msun]
cluster_z = 0.3   # Cluster's redshift
concentration = 4
ngals = 100000     # Number of galaxies
Delta = 200
cluster_ra = 0.0
cluster_dec = 0.0

ideal_data = mock.generate_galaxy_catalog(cluster_m, cluster_z, concentration, cosmo, 0.8, ngals=ngals)

cluster_id = "CL_ideal"
cl1 = clmm.GalaxyCluster(cluster_id, cluster_ra, cluster_dec,
                               cluster_z, ideal_data)

theta1, g_t1, g_x1 = cl1.compute_tangential_and_cross_components(geometry="flat")

bin_edges = da.make_bins(5.2e-5, 6e-3, 15, method='evenlog10width')

profile1 = cl1.make_radial_profile("Mpc", bins=bin_edges,cosmo=cosmo)

gal_bins = binned_statistic(theta1, [theta1, g_t1], statistic='count', bins=bin_edges)[2]

# list: each entry is the bin to which each galaxy belongs

bins_gal = []
for i in range(15):
    bin_i = []
    for j in range(len(gal_bins)):
        if gal_bins[j] == i+1:
            bin_i.append(j)
    bins_gal.append(bin_i)
bins_gal

# list: each entry is a list with the galaxies belonging to the respective bin

bins_gal_theta = []
for i in range(15):
    bin_i = []
    for gal_id in bins_gal[i]:
        bin_i.append(theta1[gal_id])
    bins_gal_theta.append(bin_i)
# bins_gal_theta

# same as bins_gal, but with theta instead of the galaxy number

bins_gal_gt = []
for i in range(15):
    bin_i = []
    for gal_id in bins_gal[i]:
        bin_i.append(g_t1[gal_id])
    bins_gal_gt.append(bin_i)
# bins_gal_theta

# same as bins_gal, but with gt instead of the galaxy number

bins_gal_gx = []
for i in range(15):
    bin_i = []
    for gal_id in bins_gal[i]:
        bin_i.append(g_x1[gal_id])
    bins_gal_gx.append(bin_i)
# bins_gal_theta

# same as bins_gal, but with gx instead of the galaxy number

for i in range(11,15):
    epdf_gt = Ncm.StatsDist1dEPDF.new_full (2000, Ncm.StatsDist1dEPDFBw.AUTO, 0.01, 0.001)
    epdf_rot_gt = Ncm.StatsDist1dEPDF.new_full (2000, Ncm.StatsDist1dEPDFBw.ROT, 0.01,  0.001)

    for gt in bins_gal_gt[i]:
        epdf_gt.add_obs(gt)
        epdf_rot_gt.add_obs(gt)

    epdf_gt.prepare()
    epdf_rot_gt.prepare()

for i in range(11,15):
    epdf_gx = Ncm.StatsDist1dEPDF.new_full (2000, Ncm.StatsDist1dEPDFBw.AUTO, 0.1, 0.001)
    epdf_rot_gx = Ncm.StatsDist1dEPDF.new_full (2000, Ncm.StatsDist1dEPDFBw.ROT, 0.1,  0.001)

    for gx in bins_gal_gx[i]:
        epdf_gx.add_obs(gx)
        epdf_rot_gx.add_obs(gx)

    epdf_gx.prepare()
    epdf_rot_gx.prepare()


scale = 1

for i in range(11,15):
    epdf_theta = Ncm.StatsDist1dEPDF.new_full (2000, Ncm.StatsDist1dEPDFBw.AUTO, 1, 0.001)
    epdf_rot_theta = Ncm.StatsDist1dEPDF.new_full (2000, Ncm.StatsDist1dEPDFBw.ROT, 1,  0.001)

    for th in bins_gal_theta[i]:
        epdf_theta.add_obs(th*scale)
        epdf_rot_theta.add_obs(th*scale)

    epdf_theta.prepare()
    epdf_rot_theta.prepare()