#!/usr/bin/env python
#
# Uniform halfspace, flat-earth location code
#
# Invert for all source parameters
#

# Externals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# Personals
from Metropolis import *


# -------------------------------------------
def verify(M,prior_bounds):
    '''
    Verify Model
    '''

    # Check stuff
    for i in range(M.size):
        if M[i]<prior_bounds[i,0] or M[i]>prior_bounds[i,1]:
            return False

    # All done
    return True


# -------------------------------------------
def calcLLK(M,data_dict):
    '''
    Compute Log likelihood
    Args:
    * M: Model
    * data_dict: Input data
    '''

    # Parse input
    x    = data_dict['x']
    vp    = data_dict['vp']
    vs    = data_dict['vs']
    data = data_dict['data']
    sigcov  = data_dict['sigma']

    # Normalization factor
    Norm = -(np.log(np.linalg.det(sigcov))+0.5*np.log(2*np.pi))*data.size

    # Residual vector
    res  = (data - calcPred(M, data_dict)) # residuals

    # Log-Likelihood
    logLLK = -0.5*(np.dot(np.dot(res.T, np.linalg.pinv(sigcov)),res)) # log of model likelihood

    # All done
    return logLLK


# -------------------------------------------
def calcPred(M, data_dict):
    '''
    Compute arrival times
    Args:
    * M: Model M[0] = origin time, M[1] = x, M[2] = y, M[3] = z, M[4] = vp
    * x1: receiver locations [(x, y, z)]
    '''
    x1 = data_dict['x']
    vp = data_dict['vp']
    vs = data_dict['vs']

    nsta, foo = x1.shape
    ttimes = np.empty(nsta*2, dtype=float)

    x = M[1:4]
    for i in xrange(nsta):
        d = np.sqrt((x1[i, 0]-x[0])**2 + (x1[i, 1]-x[1])**2 +
                    (x1[i, 2]-x[2])**2) 
        ttimes[i] = M[0] + d/vp
        ttimes[i+nsta] = M[0] + d/vs

    assert not np.isnan(ttimes).any(), 'NaN value in travel-times'

    # All done
    return ttimes


def plot_cov(cov, title=''):
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(111)
    im = ax.matshow(cov)
    plt.colorbar(im)
    plt.suptitle(title)

# -------------------------------------------
def plot_result(M,mtarget,i1,i2,limits,names,show_ini=True,title = None):
    '''
    Plot results
    Args:
    * M: sample set
    * show_ini: Show initial sample (black star)
    * mtarget
    '''
    xbounds = limits[i1]
    ybounds = limits[i2]

    xbins = np.linspace(xbounds[0],xbounds[1],250)
    ybins = np.linspace(ybounds[0],ybounds[1],250)

    ### Slip and slip depth histograms
    H, x, y = np.histogram2d(M[:,i1],M[:,i2],bins=(xbins,ybins),normed = True)
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    Hmask = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero

    # Figure and axis objects
    figM = plt.figure(facecolor='w')

    gs1 = gridspec.GridSpec(2, 2,width_ratios=[4,1],height_ratios=[4,1])
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])
    ax3 = plt.subplot(gs1[2])

    # 2D Hist
    im = ax1.pcolormesh(x,y,Hmask,zorder=0,vmin=0)#,vmax=0.75*Cm)
    if show_ini:
        ax1.plot(M[0,i1],M[0,i2],'k*',markersize=10,zorder=2)
    ax1.plot(mtarget[i1],mtarget[i2],'r*',markersize=10,zorder=2)
    ax1.set_xlabel(names[i1])
    # ax1.xaxis.set_label_coords(0.5, -0.06)
    ax1.set_ylabel(names[i2])    
    ax1.set_ylim(ybounds[0],ybounds[1])
    ax1.set_xlim(xbounds[0],xbounds[1])
    ax1.invert_yaxis()

    ## Slip depth marginal distribution
    bins = np.linspace(ybounds[0],ybounds[1],50)
    ax2.xaxis.set_ticklabels('')
    ax2.yaxis.set_ticklabels('')
    ax2.hist(M[:,i2],bins=bins,range=(ybounds[0],ybounds[1]),normed=True,orientation='horizontal')
    ax2.plot([ax2.get_xlim()[0],ax2.get_xlim()[1]],[mtarget[i2],mtarget[i2]],'r',lw=4)
    ax2.set_ylim(ybounds[0],ybounds[1])
    ax2.invert_yaxis()        

    ## Slip marginal distribution
    bins = np.linspace(xbounds[0],xbounds[1],50)
    ax3.xaxis.set_ticklabels('')
    ax3.yaxis.set_ticklabels('')
    ax3.hist(M[:,i1],bins=bins,range=(xbounds[0],xbounds[1]),normed=True)
    ax3.plot([mtarget[i1],mtarget[i1]],[ax3.get_ylim()[0],ax3.get_ylim()[1]],'r',lw=4)
    ax3.set_xlim(xbounds[0],xbounds[1])
    ax3.invert_yaxis()

    # Title
    if title is not None:
        plt.suptitle(title)    


def calc_model_cov(M, data_dict):
    # assume a gaussian distribution around Vp and Vs
    n_samples = 1000
    n_data = len(data_dict['data'])
    vp = data_dict['vp']
    vs = data_dict['vs']
    sig_vp = data_dict['sig_vp']
    sig_vs = data_dict['sig_vs']

    vp = t_vp + sig_vp*np.random.randn(n_samples)
    vs = t_vs + sig_vs*np.random.randn(n_samples)

    all_data = np.empty((n_samples, n_data), dtype=float)

    mod_dict = {}
    for i in xrange(n_samples):
        mod_dict['x'] = data_dict['x']
        mod_dict['vp'] = vp[i]
        mod_dict['vs'] = vs[i]
        all_data[i, :] = calcPred(M, mod_dict)

    std_data = np.std(all_data, axis=0)
    
    return np.diag(std_data*std_data)

### Define input parameters ###
##############################


## Number of samples 
n_samples = 50000


## Target model (Actual "true" solution) 
names = ['Origin time', 'x (km)', 'y (km)', 'z (km)', 'Vp (km/s)', 'Vs (km/s)']

# origin time
t_otime     = 2.

# x, y z
t_x = 0.0
t_y = 0.0
t_z = 5.0

t_vp = 5.5
t_vs = t_vp/np.sqrt(3)

# Data error 
sigdata_p = 0.03
sigdata_s = 0.08

# Target model vector
mtarget = np.array([t_otime,t_x, t_y, t_z])


## Prior Bounds (assuming uniform prior) 
prior_bounds = np.array([[0.0, 4.0],  # otime
                         [-5.0, 5.0],# x
                         [-5.0, 5.0], # y
                         [0.001, 10.0]]) # z



## Covariance of the proposal PDF (Gaussian PDF)
prop_sigma = np.array([0.03, 0.03, 0.03, 0.03])
prop_cov   = np.diag(prop_sigma*prop_sigma)


## Calculate synthetic data

# Receiver locations
nsta = 10
x_sta = np.random.uniform(prior_bounds[1][0], prior_bounds[1][1], nsta)
y_sta = np.random.uniform(prior_bounds[1][0], prior_bounds[1][1], nsta)
z_sta = np.zeros(nsta)

x = np.array(zip(x_sta, y_sta, z_sta))
    
# Creation of noise free synthetic data
data_dict = {'x':x, 'vp':t_vp, 'vs':t_vs}
pred = calcPred(mtarget, data_dict)       # Unnormalized noise free data


# Noisy data
npts = len(pred)/2
sigdata = np.empty(2*npts, dtype=float)
sigdata[0:npts] = sigdata_p
sigdata[npts:] = sigdata_s
cov_data = np.diag(sigdata*sigdata)
data = pred + sigdata*np.random.randn(pred.size) # Noisy data
data_dict['data'] = data

# Creation of model discrepancy covariance
sig_vp = 0.05*t_vp
sig_vs = 0.05*t_vs
data_dict['sig_vp'] = sig_vp
data_dict['sig_vs'] = sig_vs
cov_model = calc_model_cov(mtarget, data_dict)

# Add the model discrepancy covariance and the data covariance
data_dict['sigma'] = cov_data+cov_model
plot_cov(cov_data, 'Data covariance')
plot_cov(cov_model, 'Model discrepancy covariance')
plot_cov(data_dict['sigma'], 'Full covariance')

## First sample in the markov chain 
ifirst = np.argmin(data)
xfirst = x[ifirst, :]
m_ini = np.array([np.min(data), xfirst[0], xfirst[1], xfirst[2]])


## Run Metropolis

start_time = time.time()
M, LLK, accepted = Metropolis(n_samples, calcLLK, verify, data_dict, m_ini,
                              prior_bounds, prop_cov)
run_time = time.time() - start_time

# Burn in
iburn = np.where(LLK>=LLK[n_samples/2:].mean())[0][0]

# Mean/STD
M_mean = M[iburn:,:].mean(axis=0)
M_std  = M[iburn:,:].std(axis=0)


## Output display & figures

# Print information
print("--- %s seconds ---" % (run_time))
print("Acceptance rate : %f" %(float(accepted)/n_samples))
print "Posterior mean : ", M_mean
print "2-sigma error  : ", 2*M_std

# Plot data
pred = calcPred(M_mean, data_dict)
plt.figure(facecolor='w')
ax=plt.subplot(111)
ax.errorbar(np.arange(len(data)), data, yerr=sigdata, fmt='bx')
ax.plot(pred,'ro')
ax.legend(('Data','Predictions'),loc='lower right')
ax.set_xlabel('Station number')
ax.set_ylabel('Arrival time')

# Plot results without burn-in
plot_result(M[iburn:,:],mtarget,1,2,prior_bounds,names,show_ini=False,title='Halfspace location')
plot_result(M[iburn:,:],mtarget,0,3,prior_bounds,names,show_ini=False,title='Halfspace location')

plt.show()
