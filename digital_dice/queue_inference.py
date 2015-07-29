#
# Code for doing bayesian inference on queue characteristics from
# outside a building
#
# Data = number of people per day, average length of time inisde building,
# overtime (=time between doors close and last person exiting)
#
# Model : average serving time, number of personnel
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import time

from Metropolis import Metropolis
from DeliQueue import run_day
from eqloc_halfspace import plot_cov, plot_result


def verify(M, prior_bounds):
    '''
    Verify Model
    '''

    # Check stuff
    for i in xrange(M.size):
        if M[i] < prior_bounds[i, 0] or M[i] > prior_bounds[i, 1]:
            return False

    # All done
    return True


def calcLLK(M, data_dict):
    '''
    Compute Log likelihood
    Args:
    * M: Model
    * data_dict: Input data and physics terms
    '''

    # Parse input
    data = data_dict['data']  # All P times followed by all S times
    sigcov = data_dict['sigma']  # Full data covariance matrix

    # Normalization factor
    Norm = -(np.log(np.linalg.det(sigcov)) + 0.5*np.log(2*np.pi))*data.size

    # Residual vector
    res = (data - calcPred(M, data_dict))

    # Log-Likelihood
    logLLK = Norm - 0.5*(np.dot(np.dot(res.T, np.linalg.pinv(sigcov)), res))

    # All done
    return logLLK




def calcPred(M, data_dict):
    
    average_service_time = M[0]
    n_servers = np.int(np.round(M[1]))

    lam = data_dict['entry_interval']
    tol_time = data_dict['tol_time']
    time_limit = data_dict['time_limit']

    time_step = 30.0

    pred_data = run_day(n_servers, lam, average_service_time, tol_time,
                        time_limit, time_step, outside=True)

    return pred_data


def calc_model_cov(M, data_dict):

    # assume a gaussian distribution around fixed parameters in model
    n_samples = 10
    n_data = 3
    all_data = np.empty((n_samples, n_data), dtype=float)

    # set up the sampling over the 'fixed' parameters
    entry_interval = data_dict['entry_interval'] +\
                     data_dict['sig_entry_interval'] *\
                     np.random.randn(n_samples)
    tol_time = data_dict['tol_time'] + data_dict['sig_tol_time'] *\
                     np.random.randn(n_samples)
    time_limit = data_dict['time_limit'] + data_dict['sig_time_limit'] *\
                     np.random.randn(n_samples)

    # sample the data space
    mod_dict={}
    for i in xrange(n_samples):
        mod_dict['entry_interval'] = entry_interval[i]
        mod_dict['tol_time'] = tol_time[i]
        mod_dict['time_limit'] = time_limit[i]
        all_data[i, :] = calcPred(M, mod_dict)

    # get the standard deviation of the data
    std_data = np.std(all_data, axis=0)

    # return the model parameter-induced covariance matrix
    return np.diag(std_data*std_data)


if __name__ == '__main__':

    np.random.seed()
    n_samples = 5000

    # target model
    # 0 = average service time
    # 1 = number of servers
    mtarget = [25*60., 2.]
    names = ['Service time', 'Number of servers']

    # fixed parameters (all in seconds)
    data_dict = {}
    data_dict['entry_interval'] = 15. * 60
    data_dict['tol_time'] = 60. * 60
    data_dict['time_limit'] = 10. * 3600
    data_dict['sig_entry_interval'] = 1. * 60
    data_dict['sig_tol_time'] = 5. * 60
    data_dict['sig_time_limit'] = 5. * 60

    # prior bounds on model space
    prior_bounds = np.array([[20.*60, 30.*60], [1., 3.]])
    # set a random initial model
    # m_ini = np.empty(len(mtarget), dtype=float)
    # m_ini = mtarget[:]
    m_ini = np.array([uniform.rvs(loc=20*60., scale=(30-20)*60),
                      uniform.rvs(loc=1, scale=3-1)])

    # covariance of the proposal PDF 
    prop_sigma = np.array([1.0 * 60, 0.5])
    prop_cov = np.diag(prop_sigma * prop_sigma)

    # measurement uncertainty on data
    sig_n_people = 0.1  # people
    sig_av_time = 10.0  # seconds
    sig_overtime = 10.0 # seconds
    sigdata = np.array([sig_n_people, sig_av_time, sig_overtime])
    cov_data = np.diag(sigdata*sigdata)

    # noise-free data
    pred = calcPred(mtarget, data_dict)

    # add noise
    data = pred + sigdata*np.random.randn(len(pred))
    data_dict['data'] = data

    # get covariance on data from model
    print 'Getting model discrepancy covariance'
    cov_model = calc_model_cov(mtarget, data_dict)

    data_dict['sigma'] = cov_data + cov_model
    print cov_data
    print cov_model
    # plot_cov(cov_data, 'Data covariance from measurement uncertainty')
    # plot_cov(cov_model, 'Data covariance from velocity model uncertainty')
    # plot_cov(data_dict['sigma'], 'Full data covariance')

    # Run Metropolis
    print 'Running metropolis'
    start_time = time.time()
    M, LLK, accepted = Metropolis(n_samples, calcLLK, verify, data_dict, m_ini,
                                  prior_bounds, prop_cov)
    run_time = time.time() - start_time

    # Burn in
    iburn = np.where(LLK >= LLK[n_samples/2:].mean())[0][0]

    # Mean/STD
    M_mean = M[iburn:, :].mean(axis=0)
    M_std = M[iburn:, :].std(axis=0)

    # Print information
    print("--- %s seconds ---" % (run_time))
    print("Acceptance rate : %f" % (float(accepted)/n_samples))
    print "True values : ", mtarget
    print "Posterior mean : ", M_mean
    print "2-sigma error  : ", 2*M_std

    # Plot results 
    plot_result(M[iburn:, :], 0, 1, prior_bounds, names, data_dict,
                mtarget=mtarget, title='Outside prediction')

    # show all plots
    plt.show()
