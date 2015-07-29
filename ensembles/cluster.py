import numpy as np
from sklearn.cluster import KMeans, MeanShift
from gaussian import generate_gaussian_mixture
from time import time


def get_cluster_centers_and_inertia(data, n_clust):

    cls = KMeans(n_clusters=n_clust)
    cls.fit(data)

    return cls.cluster_centers_, cls.inertia_

def get_cluster_centers(data):
    cls = MeanShift(bin_seeding=True)
    cls.fit(data)

    return cls.cluster_centers_ 

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # set up a four family game
    means = [0., -5., +5., +9]
    sigmas = [0.6, 0.8, 0.5, 1.5]
    nsamples = [100, 800, 500, 5000]

    # generate the samples
    samp = generate_gaussian_mixture(means, sigmas, nsamples)


    # Turn into a fake 2D set for clustering to work
    nsamp = len(samp)
    samp2d = np.vstack((samp, samp)).T

    # do clustering
    centers = get_cluster_centers(samp2d)
    n_cent, n_dim = centers.shape

#    # do timing according to sample size
#    t = []
#    n = []
#    nc = []
#    for mult in [1, 2, 5]:
#        # set the number of samples
#        nsamp = nsamples*mult
#        n.append(np.sum(nsamp))
#        # generate the data
#        s = generate_gaussian_mixture(means, sigmas, nsamp)
#        samp2d = np.vstack((s, s)).T
#        # start timer
#        t_start = time()
#        c = get_cluster_centers(samp2d)
#        t_end = time()
#        nclusters, n_dim = c.shape
#        nc.append(nclusters)
#        # add time to time array
#        t.append(t_end-t_start)

    # do plotting
    fig, ax = plt.subplots(1,1)
    ax.hist(samp, bins=50, normed=True, histtype='stepfilled', alpha=0.2)
    ax.plot(centers[:,0], np.zeros(n_cent)+0.05, 'o')
#    ax[1].semilogx(n,t, 'o')
#    plt.xlabel('N samples')
#    plt.ylabel('Time')
#    ax[1].semilogx(n,nc, 'o')
#    plt.xlabel('N samples')
#    plt.ylabel('N clusters')

    plt.show()

