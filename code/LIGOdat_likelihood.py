import numpy as np
import pandas as pd
import kalepy as kale
import h5py
import time

import astropy.cosmology as cosmo
from astropy import units as u

from dart_board import constants as c
from dart_board import cosmo as cm


MASS_SCALE = 100
DIST_SCALE = 1000
BANDWIDTH = [1/1e2,1/1e1]
H0 = 67.9* u.km / u.s / u.Mpc    #LIGO cosmology: H0 = 67.9; Omega_m = 0.3065
OM0 = 0.3065 # oh-em-zero
cosmology = cosmo.FlatLambdaCDM(H0=H0, Om0=OM0)

def readDetectionFile(Detection_file, key):
    """hd5 -> DataFrame: reads hd5 infile with given key"""
    BBH = h5py.File(Detection_file, 'r')
    Data_df = pd.DataFrame()
    m1 = BBH[key]['m1_detector_frame_Msun']
    m2 =BBH[key]['m2_detector_frame_Msun']

    Data_df['luminosity_distance_Mpc'] = BBH[key]['luminosity_distance_Mpc']
    Data_df['q'] = m2/m1
    Data_df['Mtot_detector_frame_Msun'] = (m1 + m2)
    return Data_df

def initializeDat(Detection_file):
    start = time.time()
    LIGO_prior = readDetectionFile(Detection_file, 'prior')
    LIGO_posterior = readDetectionFile(Detection_file, 'Overall_posterior')

    LIGO_posterior_ws = findSrcMass(LIGO_posterior)

    prior_array = dfToArray(LIGO_prior)
    posterior_array = dfToArray(LIGO_posterior_ws)
    # thin_posterior = thinPosterior(posterior_array)
    # weights = findPriorWeights(prior_array, thin_posterior[0:3])
    # return thin_posterior, weights
    weights = findPriorWeights(prior_array, posterior_array[0:3])
    return posterior_array, weights

def findSrcMass(Data_df):
    global cosmology
    Dvals = (1e-4 + np.random.rand(1000000) * 2000) * u.Mpc #interpolation setup and execution
    zmin = cosmo.z_at_value(cosmology.luminosity_distance, Dvals.min())
    zmax = cosmo.z_at_value(cosmology.luminosity_distance, Dvals.max())
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
    Dgrid = cosmology.luminosity_distance(zgrid)
    zvals = np.interp(Data_df['luminosity_distance_Mpc'], Dgrid.value, zgrid)

    Data_df['Mtot_source_frame_Msun'] = Data_df['Mtot_detector_frame_Msun'] / (1+ zvals)
    Data_df['z'] = zvals
    Data_df['lookback_time(Gyr)'] = cosmology.lookback_time(zvals)
    Data_df = Data_df.drop('Mtot_detector_frame_Msun', axis = 1)
    return Data_df

def dfToArray(Data_df):
    Kale_array = Data_df.to_numpy()
    Kale_array = Kale_array.transpose() #.T?
    return Kale_array

def thinPosterior(posterior_array):
    new_posterior_array = np.transpose(posterior_array)
    inds = np.random.randint(len(new_posterior_array), size=int(len(new_posterior_array)/10))
    new_posterior_array = new_posterior_array[inds]
    posterior_sample = new_posterior_array.T
    return posterior_sample

def findPriorWeights(Kale_array, Points_array):
    global MASS_SCALE
    global DIST_SCALE
    scaled_prior = np.array([Points_array[0]/DIST_SCALE, Points_array[1], Points_array[2]/MASS_SCALE])
    scaled_points = np.array([Points_array[0]/DIST_SCALE, Points_array[1], Points_array[2]/MASS_SCALE])
    total_weights=np.zeros(len(Points_array))

    MAX_LUMDIST = np.max(Kale_array[0])
    REFLECT = [[0,MAX_LUMDIST],
               [0,1],
               None]
    points_posterior, density_prior = kale.density(scaled_prior, probability=True, reflect=REFLECT,
                                                  points = scaled_points)
    weights = 1/density_prior
    return weights

def findLikelihoodM(Kale_array, Points_array):
    global BANDWIDTH
    ndim = 2
    kde=np.ndarray((ndim,len(Kale_array[0])))
    reflected_kde= np.ndarray((ndim,len(Kale_array[0])))

    reflect_array = np.array([2 - Kale_array[0], #reflect over q = 1
                                     Kale_array[1]])
    reflect_points = np.array([2 - Points_array[0], #reflect over q = 1
                                     Points_array[1]])
    i = 0
    while i < ndim:
        kde[i] = np.exp(-((Kale_array[i] - Points_array[i])**2)/(2*BANDWIDTH[i]**2))
        i += 1
    i=0
    while i < ndim:
        reflected_kde[i] = np.exp(-((reflect_array[i] - reflect_points[i])**2)/(2*BANDWIDTH[i]**2))
        i += 1

    total_likelihood = np.concatenate((kde,reflected_kde), axis = 1)
    return total_likelihood

def findLikelihoodD(posterior_array, t_corr, Z):
    global cosmology
    Gyr_to_Myr = 1000
    z = cm.utilities.get_z_from_t(Gyr_to_Myr*(posterior_array[4]+ t_corr))
    diff_vol_corr = 4*np.pi*(cosmology.comoving_distance(z)**2)
    p_sfh = cm.utilities.calc_SFR(z)
    ln_p_z = np.zeros(len(posterior_array[0]))
    ln_p_z = cm.metallicity.ln_prior_z(np.log(Z), #ln_Z_b
                                        np.log((posterior_array[4]+t_corr)*Gyr_to_Myr), #ln_t_b, ln(Myr)
                                        z_min=c.min_z, z_max=c.max_z)
    return p_sfh*(1/(u.Mpc**2))*np.exp(ln_p_z)*diff_vol_corr / (1+z)
