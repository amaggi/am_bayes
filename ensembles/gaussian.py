import numpy as np
from scipy.stats import norm


def generate_gaussian_mixture(means, sigmas, nsamples):
    n = len(means)
    r = []
    for i in xrange(n):
        r.append(norm.rvs(loc=means[i], scale=sigmas[i], size=nsamples[i]))

    return np.concatenate(r)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    means = [0., -5., +5.]
    sigmas = [0.6, 0.8, 0.2]
    nsamples = [1000, 2000, 1000]

    samp = generate_gaussian_mixture(means, sigmas, nsamples)
    fig, ax = plt.subplots(1,1)
    ax.hist(samp, bins=50, normed=True, histtype='stepfilled', alpha=0.2)
    plt.show()


