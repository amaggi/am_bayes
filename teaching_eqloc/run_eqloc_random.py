import time
import numpy as np
import matplotlib.pyplot as plt

from Metropolis import Metropolis
from eqloc_halfspace import deploy_random_stations, calcPred, calc_model_cov,\
     plot_cov, calcLLK, verify, plot_datafit, plot_result, deploy_circular_array


# Number of samples 
n_samples = 50000

# Target model (Actual "true" solution) 
t_vp = 5.5
t_vs = t_vp/np.sqrt(3)

# origin time
t_otime = 2.

# x, y z
t_x = 0.0
t_y = 0.0
t_z = 5.0

# set up the target vector
mtarget = np.array([t_otime,t_x, t_y, t_z])
names = ['Origin time', 'x (km)', 'y (km)', 'z (km)']

# measurement uncertainty
sigdata_p = 0.01
sigdata_s = 0.05

# velocity model uncertainty
sig_vp = 0.03 * t_vp
sig_vs = 0.03 * t_vs

# number of stations
nsta = 10

# Prior Bounds (assuming uniform prior) 
prior_bounds = np.array([[0.0, 4.0],  # otime
                         [-5.0, 5.0],# x
                         [-5.0, 5.0], # y
                         [0.001, 10.0]]) # z

# Covariance of the proposal PDF (Gaussian PDF)
prop_sigma = np.array([0.03, 0.03, 0.03, 0.03])
prop_cov   = np.diag(prop_sigma*prop_sigma)

# station positions
# x = deploy_random_stations(nsta, prior_bounds, borehole=True)
x = deploy_circular_array(nsta, prior_bounds, 3.0, 3.0, 2., borehole=False)
   
# Creation of noise free synthetic data
data_dict = {'x':x, 'vp':t_vp, 'vs':t_vs}
pred = calcPred(mtarget, data_dict)

# Set up the measurement noise
# Note that P and S-wave uncertainties are different
npts = len(pred)/2
sigdata = np.empty(2*npts, dtype=float)
sigdata[0:npts] = sigdata_p
sigdata[npts:] = sigdata_s
cov_data = np.diag(sigdata*sigdata)

# add measurement noise to the data
data = pred + sigdata*np.random.randn(pred.size) # Noisy data
data_dict['data'] = data

# Creation of model discrepancy covariance
data_dict['sig_vp'] = sig_vp
data_dict['sig_vs'] = sig_vs
cov_model = calc_model_cov(mtarget, data_dict)

# Add the model discrepancy covariance and the data covariance
data_dict['sigma'] = cov_data+cov_model
plot_cov(cov_data, 'Data covariance')
plot_cov(cov_model, 'Model discrepancy covariance')
plot_cov(data_dict['sigma'], 'Full covariance')

# Set first point in the Markov chain to the the information at the first
# station receiving the information about the event
ifirst = np.argmin(data)
xfirst = x[ifirst, :]
m_ini = np.array([np.min(data), xfirst[0], xfirst[1], xfirst[2]])

# Run Metropolis
start_time = time.time()
M, LLK, accepted = Metropolis(n_samples, calcLLK, verify, data_dict, m_ini,
                              prior_bounds, prop_cov)
run_time = time.time() - start_time

# Burn in
iburn = np.where(LLK >= LLK[n_samples/2:].mean())[0][0]

# Mean/STD
M_mean = M[iburn:, :].mean(axis=0)
M_std = M[iburn:, :].std(axis=0)


## Output display & figures

# Print information
print("--- %s seconds ---" % (run_time))
print("Acceptance rate : %f" % (float(accepted)/n_samples))
print "Posterior mean : ", M_mean
print "2-sigma error  : ", 2*M_std

# Plot results 
plot_datafit(M_mean, data_dict)
plot_result(M[iburn:, :], 1, 2, prior_bounds, names, data_dict, plot_sta=True,
            mtarget=mtarget, title='Halfspace location')
plot_result(M[iburn:, :], 0, 3, prior_bounds, names, data_dict, mtarget=mtarget,
            title='Halfspace location')

# show all plots
plt.show()
