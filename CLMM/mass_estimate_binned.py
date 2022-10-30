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
shapenoise    = [0.2, 0.05, 0.01]  # True ellipticity standard variation
# shapenoise    = [0.05]  # True ellipticity standard variation

# Create galaxy catalog and Cluster object

class GaussGammaTErr (Ncm.DataGaussDiag):
    z_cluster = GObject.Property (type = float, flags = GObject.PARAM_READWRITE)
    z_source  = GObject.Property (type = Ncm.Vector, flags = GObject.PARAM_READWRITE)
    r_source  = GObject.Property (type = Ncm.Vector, flags = GObject.PARAM_READWRITE)
    z_err     = GObject.Property (type = Ncm.Vector, flags = GObject.PARAM_READWRITE)

    def __init__ (self):
        Ncm.DataGaussDiag.__init__ (self, n_points = 0)        
        self.moo = clmm.Modeling ()
    
    def init_from_data (self, z_cluster, r_source, z_source, gt_profile, gt_err, z_err = None, moo = None):
        
        if moo:
            self.moo = moo
        
        assert len (gt_profile) == len (z_source)
        assert len (gt_profile) == len (r_source)
        assert len (gt_profile) == len (gt_err)
        
        self.set_size (len (gt_profile))
        
        self.props.z_cluster = z_cluster
        self.props.z_source  = Ncm.Vector.new_array (z_source)
        self.props.r_source  = Ncm.Vector.new_array (r_source)
        if z_err:
            self.props.r_source  = Ncm.Vector.new_array (z_err)
                
        self.y.set_array (gt_profile)
        
        self.sigma.set_array (gt_err) # Diagonal covariance matrix: standard deviation values in gt_err.

        
        self.set_init (True)        
    
    # Once the NcmDataGaussDiag is initialized, its parent class variable np is set with the n_points value.
    def do_get_length (self):
        return self.np

    def do_get_dof (self):
        return self.np

    def do_begin (self):
        pass

    def do_prepare (self, mset):
        self.moo.set_mset (mset)
        
    def do_mean_func (self, mset, vp):
        vp.set_array (self.moo.eval_reduced_tangential_shear (self.props.r_source.dup_array (), self.props.z_cluster, self.props.z_source.dup_array ()))
        return

GObject.type_register (GaussGammaTErr)


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
    gc.make_radial_profile("Mpc", da.make_bins(0.7, 4, 30), cosmo=cosmo)

    ggt = GaussGammaTErr ()
    ggt.init_from_data (z_cluster = cluster_z, r_source = gc.profile['radius'], z_source = gc.profile['z'], gt_profile = gc.profile['gt'], gt_err = gc.profile['gt_err'], moo = moo)

    dset = Ncm.Dataset.new ()
    dset.append_data (ggt)
    lh = Ncm.Likelihood.new (dset)

    fit = Ncm.Fit.new (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)
    fit.run (Ncm.FitRunMsgs.SIMPLE)
    fit.obs_fisher ()
    fit.log_info ()
    fit.log_covar ()

    print(10**mset.param_get(MDelta_pi.mid, MDelta_pi.pid))

    Ncm.func_eval_set_max_threads (4)
    Ncm.func_eval_log_pool_stats ()

    init_sampler = Ncm.MSetTransKernGauss.new (0)
    init_sampler.set_mset (mset)
    init_sampler.set_prior_from_mset ()
    init_sampler.set_cov_from_rescale (1.0e-1)

    nwalkers = 200
    stretch = Ncm.FitESMCMCWalkerAPES.new (nwalkers, mset.fparams_len ())
    esmcmc  = Ncm.FitESMCMC.new (fit, nwalkers, init_sampler, stretch, Ncm.FitRunMsgs.SIMPLE)
    esmcmc.set_data_file (f"Fits/CLMM_binned_{sn}.fits")
    esmcmc.set_auto_trim_div (100)
    esmcmc.set_max_runs_time (2.0 * 60.0)
    esmcmc.set_nthreads(4)
    esmcmc.start_run()
    esmcmc.run(10000/nwalkers)
    esmcmc.end_run()

    mcat = esmcmc.peek_catalog()
    rows = np.array([mcat.peek_row(i).dup_array() for i in range(nwalkers * 10, mcat.len())])
    params = ["$" + mcat.col_symb(i) + "$" for i in range (mcat.ncols())]

    # partial = ChainConsumer()
    # partial.add_chain(rows[:,1:], parameters=params[1:], name=f"$\sigma_{{\epsilon^s}} = {sn}$")
    # partial.configure(spacing=0.0, usetex=True, colors='#D62728', shade=True, shade_alpha=0.2, bar_shade=True, smooth=True, kde=True, legend_color_text=False, linewidths=2)

    # CC_fig = partial.plotter.plot(figsize=(8, 8), truth=[4, 15])

    # fig = plt.figure(num=CC_fig, figsize=(8,8), dpi=300, facecolor="white")
    # fig.savefig(f"Plots/MCMC/KDE_lh_corner_{sn}.png")


    total.add_chain(rows[:,1:], parameters=params[1:], name=f"$\sigma_{{\epsilon^s}} = {sn}$")

total.configure(spacing=0.0, usetex=True, colors=['#D62728', '#1F77B4', "#9467BD"], shade=True, shade_alpha=0.2, bar_shade=True, label_font_size=20, smooth=[True, True, True,], kde=[True, True, True], legend_color_text=False, linewidths=[3, 2, 1], linestyles=["-", "-", "-"], sigmas=[1,2])

total_fig = total.plotter.plot(figsize=(8,8), truth=[4, 15])
fig = plt.figure(num=total_fig, figsize=(8,8), dpi=300, facecolor="white")
fig.savefig(f"Plots/MCMC/CLMM_binned_corner.png")

total_fig = total.plotter.plot(figsize=(8, 8), truth=[4, 15], chains=[f"$\sigma_{{\epsilon^s}} = {shapenoise[1]}$", f"$\sigma_{{\epsilon^s}} = {shapenoise[2]}$"])
fig = plt.figure(num=total_fig, figsize=(8,8), dpi=300, facecolor="white")
fig.savefig(f"Plots/MCMC/CLMM_binned_corner_zoom.png")
