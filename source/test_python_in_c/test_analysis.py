""" Test script to analyze one dataset with the emcee and corner packages: """

import numpy as np
import emcee
import corner
import scipy.optimize as op
from scipy.special import factorial
from matplotlib import pyplot as plt

""" Load information about the generation of the datasets from file (np.array of float): """
# TODO: Check, if info-files have the same parameter:
info_dataset = np.loadtxt("dataset_output_20/datasets/info_dataset_701_to_1000.txt")
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]

""" Load simulated spectra in events/MeV from file (np.array of float): """
file_signal = "gen_spectrum_v2/signal_DMmass20_bin100keV.txt"
Spectrum_signal = np.loadtxt(file_signal)
file_info_signal = "gen_spectrum_v2/signal_info_DMmass20_bin100keV.txt"
info_signal = np.loadtxt(file_info_signal)
file_DSNB = "gen_spectrum_v2/DSNB_EmeanNuXbar22_bin100keV.txt"
Spectrum_DSNB = np.loadtxt(file_DSNB)
file_reactor = "gen_spectrum_v2/Reactor_NH_power36_bin100keV.txt"
Spectrum_reactor = np.loadtxt(file_reactor)
file_CCatmo = "gen_spectrum_v2/CCatmo_Osc1_bin100keV.txt"
Spectrum_CCatmo = np.loadtxt(file_CCatmo)

""" Get Dark Matter mass from the info_signal file: """
DM_mass = info_signal[9]

""" Define the energy window, where spectrum of virtual experiment and simulated spectrum is analyzed
    (from min_E_cut in MeV to max_E_cut in MeV): """
# TODO-me: is it correct to just look at a energy window?
# min_E_cut = DM_mass - 5
min_E_cut = min_E_visible
# max_E_cut = DM_mass + 5
max_E_cut = max_E_visible

E_cut = np.arange(min_E_cut, max_E_cut+interval_E_visible, interval_E_visible)
# calculate the entry number of the array to define the energy window:
entry_min_E_cut = int((min_E_cut - min_E_visible) / interval_E_visible)
entry_max_E_cut = int((max_E_cut - min_E_visible) / interval_E_visible)

""" Simulated spectra in events/bin (multiply with interval_E_visible): """
# spectrum per bin in the 'interesting' energy range from min_E_cut to max_E_cut
# (you have to take (entry_max+1) to get the array, that includes max_E_cut):
spectrum_Signal_per_bin = Spectrum_signal[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible
spectrum_DSNB_per_bin = Spectrum_DSNB[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible
spectrum_CCatmo_per_bin = Spectrum_CCatmo[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible
spectrum_Reactor_per_bin = Spectrum_reactor[entry_min_E_cut: (entry_max_E_cut + 1)] * interval_E_visible


""" 'true' value of the parameter: """
N_events_signal = np.sum(spectrum_Signal_per_bin)
N_events_DSNB = np.sum(spectrum_DSNB_per_bin)
N_events_CCatmo = np.sum(spectrum_CCatmo_per_bin)
N_events_reactor = np.sum(spectrum_Reactor_per_bin)


""" Get data from the model: """
# load corresponding dataset (unit: events/bin) (np.array of float):
Data = np.loadtxt("dataset_output_20/datasets/Dataset_1000.txt")
# dataset in the 'interesting' energy range from min_E_cut to max_E_cut
# (you have to take (entry_max+1) to get the array, that includes max_E_cut):
Data = Data[entry_min_E_cut: (entry_max_E_cut + 1)]


# Fraction of DM signal (np.array of float):
fraction_Signal = spectrum_Signal_per_bin / N_events_signal

# Fraction of DSNB background (np.array of float):
fraction_DSNB = spectrum_DSNB_per_bin / N_events_DSNB

# Fraction of CCatmo background (np.array of float)::
fraction_CCatmo = spectrum_CCatmo_per_bin / N_events_CCatmo

# Fraction of reactor background (np.array of float)::
fraction_Reactor = spectrum_Reactor_per_bin / N_events_reactor


def lnlikelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    s, b_dsnb, b_ccatmo, b_reactor = param
    lamb = fraction_signal*s + fraction_dsnb*b_dsnb + fraction_ccatmo*b_ccatmo + fraction_reactor*b_reactor
    sum_1 = data*np.log(lamb)
    sum_2 = np.log(factorial(data, exact=True))
    sum_3 = lamb

    return np.sum(sum_1 - sum_2 - sum_3)


negll = lambda *args: -lnlikelihood(*args)

parameter_guess = np.array([N_events_signal, N_events_DSNB, N_events_CCatmo, N_events_reactor])
bnds = ((0, None), (0, None), (0, None), (0, None))

result = op.minimize(negll, parameter_guess,
                     args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor),
                     method='L-BFGS-B', bounds=bnds)

N_signal_maxlikeli, N_dsnb_maxlikeli, N_ccatmo_maxlikeli, N_reactor_maxlikeli = result["x"]

print("signal best-fit = {0}".format(N_signal_maxlikeli))
print("DSNB background best-fit = {0}".format(N_dsnb_maxlikeli))
print("CCatmo background best-fit = {0}".format(N_ccatmo_maxlikeli))
print("Reactor background best-fit = {0}".format(N_reactor_maxlikeli))


def lnprior(param):
    s, b_dsnb, b_ccatmo, b_reactor = param
    if 0 <= s <= 10:
        ln_prior_s = np.log(1/10)
    else:
        ln_prior_s = -np.inf

    if 0 <= b_dsnb <= 2*N_events_DSNB:
        ln_prior_b_dsnb = np.log(1/(2*N_events_DSNB))
    else:
        ln_prior_b_dsnb = -np.inf

    if (N_events_CCatmo - N_events_CCatmo/2) <= b_ccatmo <= (N_events_CCatmo + N_events_CCatmo/2):
        ln_prior_b_ccatmo = np.log(1/N_events_CCatmo)
    else:
        ln_prior_b_ccatmo = -np.inf

    if (N_events_reactor - N_events_reactor/2) <= b_reactor <= (N_events_reactor + N_events_reactor/2):
        ln_prior_b_reactor = np.log(1/N_events_reactor)
    else:
        ln_prior_b_reactor = -np.inf

    return ln_prior_s + ln_prior_b_dsnb + ln_prior_b_ccatmo + ln_prior_b_reactor


def lnpostprob(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor):
    lnp = lnprior(param)

    if not np.isfinite(lnp):
        return -np.inf

    return lnp + lnlikelihood(param, data, fraction_signal, fraction_dsnb, fraction_ccatmo, fraction_reactor)




""" After all this setup, it’s easy to sample this distribution using emcee. We’ll start by initializing the 
    walkers in a tiny Gaussian ball around the maximum likelihood result (I’ve found that this tends to be a 
    pretty good initialization in most cases): """
ndim, nwalkers = 4, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

""" Then, we can set up the sampler: """
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpostprob,
                                args=(Data, fraction_Signal, fraction_DSNB, fraction_CCatmo, fraction_Reactor))

""" and run the MCMC for 500 steps starting from the tiny ball defined above: """
sampler.run_mcmc(pos, 500)

""" We can plot these traces out to see what's happening: """
# fig1 = plt.plot(sampler.chain[:, :, 0].T, '-', color='k', alpha=0.3)
# plt.show()


""" we’ll just accept it and discard the initial 50 steps and flatten the chain so that we have a flat list of 
    samples: """
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

""" Now that we have this list of samples, let’s make one of the most useful plots you can make with your MCMC 
    results: a corner plot. Generate a corner plot is as simple as: """
fig = corner.corner(samples, labels=["$S$", "$B_{DSNB}$", "$B_{CCatmo}$", "$B_{reactor}$"],
                    truths=[N_events_signal, N_events_DSNB, N_events_CCatmo, N_events_reactor],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, labels_args={"fontsize": 40})
fig.savefig("test_analysis.png")

"""
samples[:, 2] = np.exp(samples[:, 2])
S, B_DSNB, B_CCatmo, B_Reactor = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                                     zip(*np.percentile(samples, [16, 50, 90], axis=0)))

print(S)
print(B_DSNB)
print(B_CCatmo)
print(B_Reactor)
"""
