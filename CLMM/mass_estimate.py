import os
import sys
import gi

gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import GObject
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

os.environ['CLMM_MODELING_BACKEND'] = 'nc'

__name__ = "NcContext"

Ncm.cfg_init ()
Ncm.cfg_set_log_handler (lambda msg: sys.stdout.write (msg) and sys.stdout.flush ())

import clmm
import numpy as np
import time
from astropy import units
from numpy import random
import clmm.dataops as da
import clmm.theory as theory
from clmm import Cosmology
from clmm.support import mock_data as mock
from clmm.utils import convert_units
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

plt.rcParams['text.latex.preamble'] = [r'\usepackage{pxfonts, mathpazo}']
plt.rcParams['font.family']=['Palatino']
plt.rc('text', usetex=True)

np.random.seed(0)
# Define cosmological parameters
cosmo = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)
    
cluster_m     = 1.e15 # Cluster mass
cluster_z     = 0.4   # Cluster redshift
concentration = 4     # Concentrion parameter NFW profile
ngals         = 10000 # Number of galaxies
Delta         = 200   # Overdensity parameter definition NFW profile
cluster_ra    = 0.0   # Cluster right ascension
cluster_dec   = 0.0   # Cluster declination
shapenoise    = [1e-2, 1e-3]  # True ellipticity standard variation

# Create galaxy catalog and Cluster object

def create_nc_data_cluster_wl (theta, g_t, z_source, z_cluster, cosmo, dist, sigma_z = None, sigma_g = None):
    r  = clmm.convert_units (theta, "radians", "Mpc", redshift = z_cluster, cosmo = cosmo)
    ga = Ncm.ObjArray.new ()
    
    sigma_g = 1.0e-4 if not sigma_g else sigma_g
    m_obs = np.column_stack ((r, g_t, np.repeat (sigma_g, len (r))))
    
    grsg = Nc.GalaxyWLReducedShearGauss (pos = Nc.GalaxyWLReducedShearGaussPos.R)
    grsg.set_obs (Ncm.Matrix.new_array (m_obs.flatten (), 3))
    

    gzgs = Nc.GalaxyRedshiftSpec ()
    gzgs.set_z (Ncm.Vector.new_array (z_source))

    gwl  = Nc.GalaxyWL (wl_dist = grsg, gz_dist = gzgs)
    ga.add (gwl)

    nc_dcwl = Nc.DataClusterWL (galaxy_array = ga, z_cluster = z_cluster)
    nc_dcwl.set_init (True)
    
    return nc_dcwl

def create_fit_obj (data_array, mset):
    dset = Ncm.Dataset.new ()
    for data in data_array:
        dset.append_data (data)
    lh = Ncm.Likelihood.new (dset)
    fit = Ncm.Fit.new (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)

    return fit

total = ChainConsumer()

for sn in shapenoise:

    moo = clmm.Modeling (massdef='mean', delta_mdef=200, halo_profile_model='nfw')
    moo.set_cosmo(cosmo)
    mset = moo.get_mset ()

    MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
    cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

    mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
    mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
    mset.prepare_fparam_map ()

    data = mock.generate_galaxy_catalog(cluster_m, cluster_z, concentration, cosmo, "chang13", zsrc_min = cluster_z + 0.1, shapenoise=sn, ngals=ngals, cluster_ra=cluster_ra, cluster_dec=cluster_dec)
    gc = clmm.GalaxyCluster("CL_noisy_z", cluster_ra, cluster_dec, cluster_z, data)
    gc.compute_tangential_and_cross_components(geometry="flat")

    ggt = create_nc_data_cluster_wl (gc.galcat['theta'], gc.galcat['et'], gc.galcat['z'], cluster_z, cosmo, cosmo.dist, sigma_g=sn)
    fit = create_fit_obj ([ggt], mset)
    fit.run (Ncm.FitRunMsgs.FULL)
    fit.obs_fisher ()
    fit.log_info ()
    fit.log_covar ()
    Ncm.func_eval_set_max_threads (6)
    Ncm.func_eval_log_pool_stats ()

    init_sampler = Ncm.MSetTransKernGauss.new (0)
    init_sampler.set_mset (mset)
    init_sampler.set_prior_from_mset ()
    init_sampler.set_cov_from_rescale (1.0e-1)

    nwalkers = 200
    stretch = Ncm.FitESMCMCWalkerAPES.new (nwalkers, mset.fparams_len ())
    esmcmc  = Ncm.FitESMCMC.new (fit, nwalkers, init_sampler, stretch, Ncm.FitRunMsgs.FULL)
    esmcmc.set_data_file (f"Fits/KDE_lh_{sn}.fits")
    esmcmc.set_auto_trim_div (100)
    esmcmc.set_max_runs_time (2.0 * 60.0)
    esmcmc.set_nthreads(4)
    esmcmc.start_run()
    esmcmc.run(10000/nwalkers)
    esmcmc.end_run()

    mcat = esmcmc.peek_catalog()
    rows = np.array([mcat.peek_row(i).dup_array() for i in range(nwalkers * 10, mcat.len())])
    params = ["$" + mcat.col_symb(i) + "$" for i in range (mcat.ncols())]

    partial = ChainConsumer()
    partial.add_chain(rows[:,1:], parameters=params[1:], name=f"$\sigma_{{\epsilon^s}} = {sn}$", statistics="max")
    partial.configure(spacing=0.0, usetex=True, colors='#D62728', shade=True, shade_alpha=0.2, bar_shade=True, label_font_size=20, smooth=True, kde=True, legend_color_text=False, linewidths=2)

    partial.plotter.plot(filename=f"Plots/MCMC/KDE_lh_corner_{sn}.png", figsize=(16, 16), truth=[4, 15])
    partial.plotter.plot(filename=f"Plots/MCMC/KDE_lh_corner_{sn}.pdf", figsize=(16, 16), truth=[4, 15])


    total.add_chain(rows[:,1:], parameters=params[1:], name=f"$\sigma_{{\epsilon^s}} = {sn}$", statistics="max")

total.configure(spacing=0.0, usetex=True, colors=['#D62728', '#1F77B4'], shade=True, shade_alpha=0.2, bar_shade=True, label_font_size=20, smooth=[True, True], kde=[True, True], legend_color_text=False, linewidths=[2, 2])
total.plotter.plot(filename=f"Plots/MCMC/KDE_lh_corner.png", figsize=(16, 16), truth=[4, 15])
total.plotter.plot(filename=f"Plots/MCMC/KDE_lh_corner.pdf", figsize=(16, 16), truth=[4, 15])
