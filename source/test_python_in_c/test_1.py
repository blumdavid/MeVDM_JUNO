""" test script for marginalization and corner plot: """

import numpy as np
import corner
import scipy.optimize as op
import emcee
from matplotlib import pyplot as plt

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)


def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


nll = lambda *args: -lnlike(*args)

result = op.minimize(nll, np.array([m_true, b_true, np.log(f_true)]), args=(x, y, yerr))

m_ml, b_ml, lnf_ml = result["x"]


def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

sampler.run_mcmc(pos, 500)


fig, (ax1, ax2, ax3) = plt.subplots(ndim, 1, sharex='all')
fig.set_size_inches(10, 10)
ax1.plot(sampler.chain[:, :, 0].T, '-', color='k', alpha=0.3)
ax1.axhline(y=m_true, color='b', linestyle='--')
ax1.set_ylabel('$m$')
ax2.plot(sampler.chain[:, :, 1].T, '-', color='k', alpha=0.3)
ax2.axhline(y=b_true, color='b', linestyle='--')
ax2.set_ylabel('$b$')
ax3.plot(sampler.chain[:, :, 2].T, '-', color='k', alpha=0.3)
ax3.axhline(y=np.log(f_true), color='b', linestyle='--')
ax3.set_ylabel('$ln(f)$')
fig.savefig('chain_traces_test1.png')


samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                    truths=[m_true, b_true, np.log(f_true)])
fig.savefig("triangle.png")

samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
