import numpy as np
import LIGOdat_likelihood as lhfile

import dart_board
from dart_board.pop_synth.cosmic_wrapper import evolve
from dart_board import constants as c

infile = '/Users/andrews/Research/dart_GW150914/data/GW150914_GWTC-1.hdf5'

prior_kwargs = {'kick_sigma' : 10}
model_kwargs = {'bhflag' : 3, 'acc_lim': 0, 'f_acc' : 0.5}
posterior_array=np.array([],dtype=np.float64) #lum_dist, q ,mtot ,z_lb , t_lb
weights=np.array([],dtype=np.float64)

def load_LIGO_dat(filename):
    global posterior_array
    global weights
    #call LIGO dat initialization
    posterior_array, weights = lhfile.initializeDat(filename)
    return

def my_function_Z(ln_z):
    if ln_z < np.log(c.min_z) or ln_z > np.log(c.max_z):
        return -np.inf
    return np.exp(ln_z)

def findTotalLikelihood(x, output): #chains, derived
    global weights
    global posterior_array
    Gyr_to_Myr = 1000
    ZMAX_LIM = 13.75 #in Gyr

    m1 = output['M1'] #in Msun
    m2 = output['M2'] #in Msun
    ecc = output['ecc']
    t_SN2 = output['t_SN2'] #in Myr
    t_gw = dart_board.utils.calc_merger_time(m1, m2, A_f = output['a'], ecc = ecc) #in Myr
    t_corr = (t_SN2 + t_gw)/Gyr_to_Myr #in Gyr
    ln_Z = x[12]
    Z = np.exp(ln_Z)

    if (m1> m2):
        q = m2/m1
    else:
        q = m1/m2
    mtot = m1+ m2

    likelihood_mq = lhfile.findLikelihoodM(posterior_array[1:3],
                                         np.array([np.array([q]),np.array([mtot])])) #dont forget to import this
    if t_corr>ZMAX_LIM:
        likelihood_d = 0
        doubled_dist_lh = 0
    else:
        likelihood_d = lhfile.findLikelihoodD(posterior_array, t_corr, Z)
        doubled_dist_lh = np.concatenate((likelihood_d,likelihood_d))

    doubled_weights = np.concatenate((weights,weights))
    likelihood = doubled_weights*doubled_dist_lh*likelihood_mq[0]*likelihood_mq[1]
    likelihood = np.sum(likelihood, axis = None)
    return likelihood


def nlog_posterior(x, dart):
    ZMAX_LIM = 13.75 #in Gyr

    empty_arr = tuple(np.zeros(17))
    likelihood = np.array([])
    global posterior_array
    global weights
    if(len(posterior_array)==0):
        global infile
        load_LIGO_dat(infile)

    lp = dart_board.priors.ln_prior(x, dart) #call priors
    if np.isinf(lp) or np.isnan(lp):  #case outside prior space
        return (-np.inf,) + empty_arr

    ll, output = dart_board.posterior.ln_likelihood(x, dart)
    if (np.isinf(ll)):  #case not BHBH
        return (-np.inf,) + empty_arr

    likelihood = findTotalLikelihood(x, output)

    if isinstance(output, np.ndarray):
            output = tuple(output)
    return (lp + np.log(likelihood),) + tuple(output)


def nlog_likelihood(x, dart):
    ZMAX_LIM = 13.75 #in Gyr

    empty_arr = tuple(np.zeros(17))
    likelihood = np.array([])
    global posterior_array
    global weights
    if(len(posterior_array)==0):
        global infile
        load_LIGO_dat(infile)

    lp = dart_board.priors.ln_prior(x, dart) #call priors
    if np.isinf(lp) or np.isnan(lp):  #case outside prior space
        return -np.inf, empty_arr

    ll, output = dart_board.posterior.ln_likelihood(x, dart)
    if (np.isinf(ll)):  #case not BHBH
        return -np.inf, empty_arr

    likelihood = findTotalLikelihood(x, output)
    if np.isinf(likelihood) or np.isnan(likelihood):
        return -np.inf, empty_arr

    if isinstance(output, np.ndarray):
            output = tuple(output)
    return np.log(likelihood), tuple(output)
