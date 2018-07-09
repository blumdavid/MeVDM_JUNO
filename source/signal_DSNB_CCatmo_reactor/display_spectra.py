""" display_spectra.py:
    in this script the datasets and the fitted spectra are displayed.
    For each dataset, the dataset and the simulated spectrum with the values from MCMC sampling are displayed.

"""

import numpy as np
from matplotlib import pyplot as plt

""" set the DM mass in MeV (float): """
DM_mass = 30

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor"

""" set the path to the datasets: """
path_dataset = path_folder + "/dataset_output_{0}/datasets_moreBkgEv/".format(DM_mass)

""" load the dataset info file: """
dataset_info_file = np.loadtxt(path_dataset + "info_dataset_1_to_5000.txt")
# bin-width of E_visible in MeV:
interval_E_vis = dataset_info_file[0]
# minimum of E_visible in MeV:
min_E_vis = dataset_info_file[1]
# maximum of E_visible in MeV:
max_E_vis = dataset_info_file[2]
# calculate E_visible in MeV:
E_vis = np.arange(min_E_vis, max_E_vis+interval_E_vis, interval_E_vis)

""" set the path to the simulated spectra: """
path_simu_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/"

# get the file of the simulated signal spectrum (events/MeV):
spectrum_signal = np.loadtxt(path_simu_spectra + "signal_DMmass{0}_bin100keV.txt".format(DM_mass))

# get the file of the simulated DSNB spectrum (events/MeV):
spectrum_DSNB = np.loadtxt(path_simu_spectra + "DSNB_EmeanNuXbar22_bin100keV_40events.txt")

# get the file of the simulated CCatmo spectrum (events/MeV):
spectrum_CCatmo = np.loadtxt(path_simu_spectra + "CCatmo_Osc1_bin100keV_102events.txt")

# get the file of the simulated reactor spectrum (events/MeV):
spectrum_reactor = np.loadtxt(path_simu_spectra + "Reactor_NH_power36_bin100keV.txt")

""" set the path to the results of the analysis (to the 'fit' values): """
file_analysis = path_folder + "/dataset_output_{0}/analysis_mcmc_moreBkgEv/".format(DM_mass)

""" set the path, where the figures should be saved: """
path_result = path_folder + "/dataset_output_{0}/result_mcmc_moreBkgEv/".format(DM_mass)

""" how many datasets and corresponding spectra should be displayed and saved: """
start_index = 1
stop_index = 100

""" from the simulated spectra calculate the spectral shape of the spectra: """
# calculate the spectra in events/bin:
spectrum_signal = spectrum_signal*interval_E_vis
spectrum_DSNB = spectrum_DSNB*interval_E_vis
spectrum_CCatmo = spectrum_CCatmo*interval_E_vis
spectrum_reactor = spectrum_reactor*interval_E_vis

# get the number of expected events from the simulated spectrum:
S_expected = np.sum(spectrum_signal)
DSNB_expected = np.sum(spectrum_DSNB)
CCatmo_expected = np.sum(spectrum_CCatmo)
reactor_expected = np.sum(spectrum_reactor)

# fractions (normalized shapes) of signal and background spectra:
fraction_signal = spectrum_signal / S_expected
fraction_DSNB = spectrum_DSNB / DSNB_expected
fraction_CCatmo = spectrum_CCatmo / CCatmo_expected
fraction_reactor = spectrum_reactor / reactor_expected

""" loop over the datasets: """
for index in np.arange(start_index, stop_index+1, 1):
    # load the dataset:
    data = np.loadtxt(path_dataset + "Dataset_{0}.txt".format(index))

    # get the file, where the results of the analysis are saved:
    analysis_result = np.loadtxt(file_analysis + "Dataset{0}_mcmc_analysis.txt".format(index))

    # load the mode of the signal contribution from the analysis results:
    S_mode = analysis_result[0]
    # load the 90%limit of the signal contribution from the analysis results:
    S_90 = analysis_result[1]
    # load the mode of the DSNB background from the analysis results:
    DSNB_mode = analysis_result[2]
    # load the mode of the CCatmo background from the analysis results:
    CCatmo_mode = analysis_result[3]
    # load the mode of the reactor background from the analysis results:
    reactor_mode = analysis_result[4]

    # total spectrum in events per bin:
    spectrum_total = (S_mode*fraction_signal + DSNB_mode*fraction_DSNB + CCatmo_mode*fraction_CCatmo +
                      reactor_mode*fraction_reactor)

    """ Test to check the number of signal events and background events in the region around the signal peak: 
        Maybe this explains the peak in the signal distribution at S=0.7
    """
    # energy range for DM mass = 30 MeV in MeV:
    E_min = 28.0
    # print("minimum E_cut = {0}".format(E_min))
    E_max = 28.6
    # print("maximum E_cut = {0}".format(E_max))

    # get the indices in the array, which correspond to E_min and E_max, respectively:
    entry_E_min = int((E_min - min_E_vis) / interval_E_vis)
    entry_E_max = int((E_max - min_E_vis) / interval_E_vis)
    # print("index of E_min = {0}, index of E_max = {1}".format(entry_E_min, entry_E_max))

    # calculate the number of events in this energy region from the spectra:
    S_cut = np.sum(S_mode * fraction_signal[entry_E_min: (entry_E_max + 1)])
    # print("number of signal events in energy region = {0}".format(S_cut))

    DSNB_cut = np.sum(DSNB_mode * fraction_DSNB[entry_E_min: (entry_E_max + 1)])
    # print("number of DSNB background events in energy region = {0}".format(DSNB_cut))

    CCatmo_cut = np.sum(CCatmo_mode * fraction_CCatmo[entry_E_min: (entry_E_max + 1)])
    # print("number of CCatmo background events in energy region = {0}".format(CCatmo_cut))

    # background events in this energy region:
    events_bkg = DSNB_cut + CCatmo_cut
    # print("number of background events in energy region = {0}".format(events_bkg))

    # total number of events in this region:
    events_total = S_cut + events_bkg
    # print("total number of events in energy region = {0}".format(events_total))

    # number of datapoints in this region:
    datapoints = np.sum(data[entry_E_min: (entry_E_max + 1)])
    # print("number of datapoints in energy region = {0}".format(datapoints))

    p1 = plt.figure(1, figsize=(15, 8))
    plt.step(E_vis, data, where='mid', color='b',
             label="data from dataset (datapoints in energy region = {0})".format(datapoints))
    plt.plot(E_vis, S_mode*fraction_signal, color='m', linestyle='--',
             label="signal spectrum with mode from analysis (S_mode={0:.4f}, S_cut={1:.4f})".format(S_mode, S_cut))
    plt.plot(E_vis, DSNB_mode*fraction_DSNB, color='g', linestyle='--',
             label="DSNB background spectrum with mode from analysis")
    plt.plot(E_vis, CCatmo_mode*fraction_CCatmo, color='r', linestyle='--',
             label="CCatmo background spectrum with mode from analysis")
    plt.plot(E_vis, reactor_mode*fraction_reactor, color='c', linestyle='--',
             label="reactor background spectrum with mode from analysis\n"
                   "(total background events B_cut={0:.4f})".format(events_bkg))
    plt.plot(E_vis, spectrum_total, color='k', label='total spectrum with results from the analysis\n'
                                                     '(total events in energy region = {0:.4f})'.format(events_total))
    plt.axvline(E_min, linestyle=':')
    plt.axvline(E_max, linestyle=':', label="energy region of signal (from {0} to {1} MeV)".format(E_min, E_max))
    plt.xlabel('Visible energy in MeV')
    plt.ylabel('number of events per bin (bin-width={0:.2f}MeV)'.format(interval_E_vis))
    plt.title('Dataset_{0} and the corresponding spectrum with the fit results from the analysis'.format(index))
    plt.ylim(ymax=3)
    plt.ylim(ymin=0.01)
    plt.xticks(np.arange(min_E_vis, max_E_vis+5, 5))
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(path_result + 'spectrum_dataset{0}'.format(index))
    plt.close(p1)




