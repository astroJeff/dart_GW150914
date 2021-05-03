import numpy as np
import dart_board
import time
import LIGOdat_likelihood as lhfile
from dart_board.pop_synth.cosmic_wrapper import evolve
from dart_board import constants as c

import mpi4py
from schwimmbad import MPIPool
# from emcee.utils import MPIPool
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

        
#infile = '/projects/b1095/jlc3821/pythonenvs/data/GW150914_GWTC-1.hdf5'
infile = '/home/jja744/Research/dart_ligo/GW150914/data/GW150914_GWTC-1.hdf5' 
prior_kwargs = {'kick_sigma' : 10}
model_kwargs = {'bhflag' : 3, 'acc_lim': 0, 'f_acc' : 0.5, 'alpha1' : 5}
posterior_array=np.array([],dtype=np.float64) #lum_dist, q ,mtot ,z_lb , t_lb
weights=np.array([],dtype=np.float64)

#START_PT = [4.02064358243423, 3.970710368090262, 8.064025311360627, 0.5865971042650004, 122.37838894004148, 1.7218207947144324, 0.31111285273303685, 1.264285671671154, 158.8843811870745, 1.2527718225062388, 0.9987209711018813, 5.7147388480851316, np.array([-3.91136795])]   #see LIGOdat_likelihood.ipynb [-26.11402514]


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
    if np.isinf(ll) or np.isnan(ll):  #case not BHBH
        return (-np.inf,) + empty_arr
    
    likelihood = findTotalLikelihood(x, output)

    if likelihood <= 0.0 or np.isnan(likelihood):
        return (-np.inf,) + empty_arr   
 
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

def findStartPt(chains, derived):
    stars = []
    ZAMS_stars = []
    max_lh = 0
    start_star = []
    flat_derived = np.reshape(derived, len(derived)*len(derived[0]))
    flat_chains = np.reshape(chains, (len(chains)*len(chains[0]),len(chains[0][0])))
    for star,ZAMS_star in zip(flat_derived,flat_chains): #broad filter on big mass range only

        if np.exp(ZAMS_star[2])*(1-ZAMS_star[3]) < 10**2.5 or ZAMS_star[3] > 0.6: continue

        Myr_to_Gyr = 1000
        HUBBLE_TIME = 1.47e4 #in Myr
        ZMAX_LIM = 13.75 #in Gyr
        MTOT_MIN = 63
        MTOT_MAX= 70
        global posterior_array
        global weights
        #likelihood = np.ndarray((3,len(weights)))
        if(len(posterior_array)==0): 
            global infile
            load_LIGO_dat(infile) 
        m1 = star['M1'] #in Msun
        m2 = star['M2'] #in Msun
        ecc = star['ecc']
        t_SN2 = star['t_SN2'] #in Myr
        #print('t_SN2 ', t_SN2)
            #t_gw = calc_merger_time(output['M1'], output['M2'], A_f=output['a'], ecc=ecc)
        t_gw = dart_board.utils.calc_merger_time(m1, m2, A_f = star['a'], ecc = ecc) #in Myr
        t_corr = (t_SN2 + t_gw)/Myr_to_Gyr #in Gyr

        mtot = m1+ m2
        if ((mtot > MTOT_MIN) & (mtot <MTOT_MAX)):
            stars.append(star)
            ZAMS_stars.append(ZAMS_star)
    for star,ZAMS_star in zip(stars,ZAMS_stars): 
        likelihood = findTotalLikelihood(ZAMS_star, star)
        if(likelihood > max_lh):
            start_star = ZAMS_star
            max_lh = likelihood
    if (len(start_star) != 0):
        global START_PT
        START_PT = np.array(start_star)
        print('starting point for second run: ', START_PT)
    else:
        print('No stars with nonzero likelihoods')
    return

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

     
    # Set up the sampler
    pub = dart_board.DartBoard("BHBH", evolve_binary=evolve, nwalkers=320, prior_kwargs=prior_kwargs, model_kwargs = model_kwargs, ln_prior_t=None, ln_prior_z=my_function_Z, ln_posterior_function=None, pool = pool)

    # Initialize the walkers
    #pub.aim_darts(N_iterations=100000)

    #start_time = time.time()

    # Run the sampler
    #pub.throw_darts(nburn=2, nsteps=20000)

    #end_time = time.time()
    #print("BHBH throw_darts finished in", end_time-start_time, "seconds.")
    
    # Save the chains
    #np.save("data/150914/BHBH_chains_alph5.npy", pub.chains)
    #np.save("data/150914/BHBH_derived_alph5.npy", pub.derived)

    pub.chains = np.load("../data/150914/BHBH_chains_alph5.npy")
    pub.derived = np.load("../data/150914/BHBH_derived_alph5.npy")

    #begin second run, starting from a point in first
    findStartPt(pub.chains, pub.derived)
    
    # Set up the sampler
    pub150914 = dart_board.DartBoard("BHBH", evolve_binary=evolve, nwalkers=320, ntemps=4, prior_kwargs=prior_kwargs, model_kwargs = model_kwargs, ln_prior_t=None, ln_prior_z=my_function_Z, ln_likelihood_function=nlog_likelihood, ln_posterior_function=nlog_posterior, pool = pool)

    # Initialize the walkers
    pub150914.aim_darts(starting_point=START_PT)

    start_time = time.time()

    # Run the sampler
    pub150914.throw_darts(nburn=2, nsteps=100000)

    end_time = time.time()
    print("throw_darts finished in", end_time-start_time, "seconds.")

    # Save the chains
    np.save("../data/150914_PT/150914_chains_alph5_long_PT.npy", pub150914.chains)
    # np.save("data/150914/150914_derived_alph5_long.npy", pub150914.derived)

