#!/usr/bin/env python
#
# Uniform halfspace, flat-earth location code
#
# Invert for all source parameters taking both measurement errors and
# model discrepancy errors into account
#

# Externals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# Personals
from Metropolis import Metropolis


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
    cov_masked = np.ma.masked_where(cov==0, cov)
    plt.figure(facecolor='w')
    ax = plt.subplot(111)
    im = ax.matshow(cov_masked)
    plt.colorbar(im)
    plt.suptitle(title)


def plot_result(M, i1, i2, limits, names, mtarget=None, title=None):
    '''
    Plot results
    Args:
    * M: sample set
    * mtarget : (true value if known)
    * i1, i2 : index of the model parameters in M to plot
    * limits : bounds on the values of all parameters (used for plot limits)
    * title : plot title
    '''

    # set bounds of figure
    xbounds = limits[i1]
    ybounds = limits[i2]

    # calculate bins for 2D histogram (to plot probability density)
    xbins = np.linspace(xbounds[0], xbounds[1], 250)
    ybins = np.linspace(ybounds[0], ybounds[1], 250)

    # 2D PDF values
    H, x, y = np.histogram2d(M[:, i1], M[:, i2], bins=(xbins, ybins),
                             normed = True)
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    Hmask = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

    # Figure and axis objects
    plt.figure(facecolor='w')

    gs1 = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])
    ax3 = plt.subplot(gs1[2])

    # 2D Hist
    ax1.pcolormesh(x, y, Hmask, zorder=0, vmin=0)
    if mtarget is not None:
        ax1.plot(mtarget[i1], mtarget[i2], 'r*', markersize=10, zorder=2)
    ax1.set_xlabel(names[i1])
    ax1.set_ylabel(names[i2])
    ax1.set_ylim(ybounds[0], ybounds[1])
    ax1.set_xlim(xbounds[0], xbounds[1])
    ax1.invert_yaxis()

    # i2 marginal distribution
    bins = np.linspace(ybounds[0], ybounds[1], 50)
    ax2.xaxis.set_ticklabels('')
    ax2.yaxis.set_ticklabels('')
    ax2.hist(M[:, i2], bins=bins, range=(ybounds[0], ybounds[1]), normed=True,
             orientation='horizontal')
    ax2.plot([ax2.get_xlim()[0], ax2.get_xlim()[1]], [mtarget[i2],
                                                      mtarget[i2]], 'r', lw=4)
    ax2.set_ylim(ybounds[0], ybounds[1])
    ax2.invert_yaxis()

    # i1 marginal distribution
    bins = np.linspace(xbounds[0], xbounds[1], 50)
    ax3.xaxis.set_ticklabels('')
    ax3.yaxis.set_ticklabels('')
    ax3.hist(M[:, i1], bins=bins, range=(xbounds[0], xbounds[1]), normed=True)
    ax3.plot([mtarget[i1], mtarget[i1]], [ax3.get_ylim()[0],
                                          ax3.get_ylim()[1]], 'r', lw=4)
    ax3.set_xlim(xbounds[0], xbounds[1])
    ax3.invert_yaxis()

    # Title
    if title is not None:
        plt.suptitle(title)


def calc_model_cov(M, data_dict):
    # assume a gaussian distribution around Vp and Vs
    n_samples = 1000
    n_data = len(data_dict['data'])
    t_vp = data_dict['vp']
    t_vs = data_dict['vs']
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


def plot_datafit(M, data_dict):

    # calculate the prediction for this model 
    pred = calcPred(M, data_dict)

    # bookkeeping
    sigdata = np.diag(data_dict['sigma'])
    data = data_dict['data']

    plt.figure(facecolor='w')
    ax = plt.subplot(111)
    ax.errorbar(np.arange(len(data)), data, yerr=np.sqrt(sigdata), fmt='bx')
    ax.plot(pred, 'ro')
    ax.legend(('Data', 'Predictions'), loc='lower right')
    ax.set_xlabel('Station number')
    ax.set_ylabel('Arrival time')


def deploy_random_stations(nsta, prior_bounds):

    nsta = 10
    x_sta = np.random.uniform(prior_bounds[1][0], prior_bounds[1][1], nsta)
    y_sta = np.random.uniform(prior_bounds[2][0], prior_bounds[2][1], nsta)
    z_sta = np.zeros(nsta)

    x = np.array(zip(x_sta, y_sta, z_sta))
    return x
