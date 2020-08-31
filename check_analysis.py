""" Script to check the Analysis:

    Display the Dataset, which was analyzed, and the simulated spectrum with the values for the number of signal
    and background events from the analysis.

    """

import numpy as np
from matplotlib import pyplot as plt

""" which dataset should be displayed: """
number_dataset = 1

""" set the path of the file that are checked: """
file_dataset = "dataset_output_20/datasets/Dataset_{0}.txt".format(number_dataset)
file_info_dataset = "dataset_output_20/datasets/info_dataset_1_to_5.txt"

file_Spectrum_signal = "gen_spectrum_v2/signal_DMmass20_bin100keV.txt"
file_Spectrum_DSNB = "gen_spectrum_v2/DSNB_EmeanNuXbar22_bin100keV.txt"
file_Spectrum_CCatmo = "gen_spectrum_v2/CCatmo_Osc1_bin100keV.txt"
file_Spectrum_Reactor = "gen_spectrum_v2/Reactor_NH_power36_bin100keV.txt"

file_analysis = "dataset_output_20/analysis_mcmc/Dataset{0}_mcmc_analysis.txt".format(number_dataset)

""" load the dataset, which was analyzed (events per bin, np.array of float): """
dataset = np.loadtxt(file_dataset)
info_dataset = np.loadtxt(file_info_dataset)
# get the bin-width of the visible energy in MeV from the info-file (float):
interval_E_visible = info_dataset[0]
# get minimum of the visible energy in MeV from info-file (float):
min_E_visible = info_dataset[1]
# get maximum of the visible energy in MeV from info-file (float):
max_E_visible = info_dataset[2]
# calculate the energy array:
energy = np.arange(min_E_visible, max_E_visible+interval_E_visible, interval_E_visible)

""" load simulated spectra (events/MeV, np.array of float): """
spectrum_signal = np.loadtxt(file_Spectrum_signal)
spectrum_DSNB = np.loadtxt(file_Spectrum_DSNB)
spectrum_CCatmo = np.loadtxt(file_Spectrum_CCatmo)
spectrum_Reactor = np.loadtxt(file_Spectrum_Reactor)

""" convert spectrum to unit event / bin: """
signal_per_bin = spectrum_signal * interval_E_visible
DSNB_per_bin = spectrum_DSNB * interval_E_visible
CCatmo_per_bin = spectrum_CCatmo * interval_E_visible
Reactor_per_bin = spectrum_Reactor * interval_E_visible

""" calculate the expected number of events from the simulated spectra: """
S_true = np.sum(signal_per_bin)
DSNB_true = np.sum(DSNB_per_bin)
CCatmo_true = np.sum(CCatmo_per_bin)
Reactor_true = np.sum(Reactor_per_bin)

""" Calculate the fraction of signal and background spectra: """
fraction_signal = signal_per_bin / S_true
fraction_DSNB = DSNB_per_bin / DSNB_true
fraction_CCatmo = CCatmo_per_bin / CCatmo_true
fraction_Reactor = Reactor_per_bin / Reactor_true

""" load the results of the analysis: """
result_analysis = np.loadtxt(file_analysis)
# Get the mean values of the number of signal and background events from the analysis:
S_50 = result_analysis[0]
DSNB_50 = result_analysis[2]
CCatmo_50 = result_analysis[3]
Reactor_50 = result_analysis[4]

""" calculate the total spectrum (event per bin) with the mean values of the analysis: """
spectrum_total = S_50*fraction_signal + DSNB_50*fraction_DSNB + CCatmo_50*fraction_CCatmo + Reactor_50*fraction_Reactor

h1 = plt.figure(1)
plt.semilogy(energy, spectrum_total, label='spectrum with the fit-values from the analysis')
plt.semilogy(energy, dataset, label='dataset')
plt.xlim([min_E_visible, max_E_visible])
plt.ylim(ymin=0)
plt.xlabel('visible energy in MeV')
plt.ylabel('counts per bin (bin={0:.2f})'.format(interval_E_visible))
plt.legend()
plt.show()

