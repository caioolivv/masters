# %%
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

try: import clmm
except:
    import notebook_install
    notebook_install.install_clmm_pipeline(upgrade=False)
    import clmm
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
from astropy import units
from numpy import random
plt.rcParams['text.latex.preamble'] = [r'\usepackage{pxfonts, mathpazo}']
plt.rcParams['font.family']=['Palatino']
plt.rc('text', usetex=True)


import clmm.dataops as da
import clmm.theory as theory
from clmm import Cosmology
from clmm.support import mock_data as mock
from clmm.utils import convert_units
from scipy.stats import chi2
from scipy.stats import kstest
import corner
import math

# %%
np.random.seed(1)
# Define cosmological parameters
cosmo = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)
    
cluster_m     = 1.e15 # Cluster mass
cluster_z     = 0.4   # Cluster redshift
concentration = 4     # Concentrion parameter NFW profile
ngals         = 10000 # Number of galaxies
Delta         = 200   # Overdensity parameter definition NFW profile
cluster_ra    = 0.0   # Cluster right ascension
cluster_dec   = 0.0   # Cluster declination
shapenoise    = 1e-1 # True ellipticity standard variation

# Create galaxy catalog and Cluster object
data = mock.generate_galaxy_catalog(cluster_m, cluster_z, concentration, cosmo, "chang13", zsrc_min = cluster_z + 0.1, shapenoise=shapenoise, ngals=ngals, cluster_ra=cluster_ra, cluster_dec=cluster_dec)
gc = clmm.GalaxyCluster("CL_noisy_z", cluster_ra, cluster_dec, cluster_z, data)

gc.compute_tangential_and_cross_components(geometry="flat")
radius = convert_units(gc.galcat['theta'], 'radians', 'Mpc', redshift=gc.z, cosmo=cosmo)

# %%
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
    #fit.set_params_reltol (1.0e-8)
    #fit.set_m2lnL_reltol (1.0e-11)
    
    return fit

ggt = create_nc_data_cluster_wl (gc.galcat['theta'], gc.galcat['et'], gc.galcat['z'], cluster_z, cosmo, cosmo.dist, sigma_g = 1e-3)

moo = clmm.Modeling (massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw')
moo.set_cosmo (cosmo)

mset = moo.get_mset ()

MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
mset.prepare_fparam_map ()


# %%
fit = create_fit_obj ([ggt], mset)

fit.run (Ncm.FitRunMsgs.SIMPLE)
fit.obs_fisher ()
fit.log_covar ()

mest = 10**mset.param_get (MDelta_pi.mid, MDelta_pi.pid)
print ("% 22.15e" % (mest))

# %%
print ("% 22.15e" % (mest))

# %%
Ncm.func_eval_set_max_threads (4)
Ncm.func_eval_log_pool_stats ()

init_sampler = Ncm.MSetTransKernGauss.new (0)
init_sampler.set_mset (mset)
init_sampler.set_prior_from_mset ()
init_sampler.set_cov_from_rescale (1.0e-1)

nwalkers = 100
stretch = Ncm.FitESMCMCWalkerAPES.new (nwalkers, mset.fparams_len ())

esmcmc  = Ncm.FitESMCMC.new (fit, nwalkers, init_sampler, stretch, Ncm.FitRunMsgs.SIMPLE)
esmcmc.set_data_file ("Data/mcmc_new_lh.fits")
esmcmc.set_auto_trim (True)
esmcmc.set_auto_trim_div (100)
esmcmc.set_max_runs_time (2.0 * 60.0)
esmcmc.set_nthreads(4)


esmcmc.start_run()
esmcmc.run(10000/nwalkers)
esmcmc.end_run()



