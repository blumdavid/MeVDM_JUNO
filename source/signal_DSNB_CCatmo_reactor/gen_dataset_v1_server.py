""" Script to generate several data-sets (virtual experiments) from the spectra,
    which are calculated with gen_spectrum_v2.py (VERSION 1) on the IHEP cluster:

    Script is based on the gen_dataset_v1.py Script, but changed a bit to be able to run it on the cluster.

    give 14 arguments to the script:
    - sys.argv[0] name of the script = gen_dataset_v1_server.py
    - sys.argv[1] Dark matter mass in MeV
    - sys.argv[2] dataset_start
    - sys.argv[3] dataset_stop
    - sys.argv[4] directory of the correct folder = "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"
    - sys.argv[5] dataset_output folder = "dataset_output_{DM_mass}"
    - sys.argv[6] datasets folder = "datasets"
    - sys.argv[7] directory of the simulated spectra = "/junofs/users/dblum/work/simu_spectra"
    - sys.argv[8] file name of DSNB spectrum = "DSNB_EmeanNuXbar22_bin100keV.txt"
    - sys.argv[9] file name of DSNB info = "DSNB_info_EmeanNuXbar22_bin100keV.txt"
    - sys.argv[10] file name of CCatmo spectrum = "CCatmo_Osc1_bin100keV.txt"
    - sys.argv[11] file name of CCatmo info = "CCatmo_info_Osc1_bin100keV.txt"
    - sys.argv[12] file name of reactor spectrum = "Reactor_NH_power36_bin100keV.txt"
    - sys.argv[13] file name of reactor info = "Reactor_info_NH_power36_bin100keV.txt"
"""

# import of the necessary packages:
import datetime
import numpy as np
import matplotlib
# To generate images without having a window appear, do:
# The easiest way is use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


def compare_4fileinputs(signal, dsnb, ccatmo, reactor, output_string):
    """
    function, which compares the input values of 4 files (signal, DSNB, CCatmo, reactor).
    If all four values are the same, the function returns the value of the signal-file.
    If one value differs from the other, the function prints a warning and returns a string.

    :param signal: value from the signal-file (float)
    :param dsnb: value from the DSNB-file (float)
    :param ccatmo: value from the CCatmo-file (float)
    :param reactor: value from the Reactor-file (float)
    :param output_string: string variable, which describes the value (e.g. 'interval_E_visible') (string)
    :return: output: either the value of the signal-file (float) or the string output_string (string)
    """

    if (signal == dsnb and ccatmo == reactor) and signal == ccatmo:
        output = signal
    else:
        output = output_string
        print("ERROR: variable {0} is not the same for the different files!".format(output))

    return output


""" get the DM mass in MeV (float): """
DM_mass = int(sys.argv[1])

""" Define parameters for saving of the virtual experiments: 
    number_dataset (dataset_stop - dataset_start) defines, how many datasets (virtual experiments)
    are generated. """
# INFO-me: 10000 datasets might be a good number for the analysis
# dataset_start defines the start point (integer):
dataset_start = int(sys.argv[2])
# dataset_stop defines the end point (integer):
dataset_stop = int(sys.argv[3])

""" Set boolean values to define, if the datasets are saved and if spectra are displayed: """
# save the dataset:
SAVE_DATA = True
### CHANGE: DISPLAY_SPECTRA is removed

""" set the path to the correct folder: """
path_folder = str(sys.argv[4])

""" set the path of the output folder: """
path_output = path_folder + "/" + str(sys.argv[5])

""" set the path of the folder, where the datasets are saved: """
path_dataset = path_output + "/" + str(sys.argv[6])

""" Files, that are read in to generate data-sets (virtual experiments): """
path_simu = str(sys.argv[7])
# TODO: check the file of the simulated signal spectrum:
file_signal = path_simu + "/signal_DMmass{0}_bin100keV.txt".format(DM_mass)
file_signal_info = path_simu + "/signal_info_DMmass{0}_bin100keV.txt".format(DM_mass)
file_DSNB = path_simu + "/" + str(sys.argv[8])
file_DSNB_info = path_simu + "/" + str(sys.argv[9])
file_CCatmo = path_simu + "/" + str(sys.argv[10])
file_CCatmo_info = path_simu + "/" + str(sys.argv[11])
file_reactor = path_simu + "/" + str(sys.argv[12])
file_reactor_info = path_simu + "/" + str(sys.argv[13])

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

# Load the calculated spectrum of the signal from the txt-file:
# load calculated spectrum from corresponding file (np.array of float):
Spectrum_signal = np.loadtxt(file_signal)
# load information about the above file (np.array of float):
info_signal = np.loadtxt(file_signal_info)
# get bin-width of E_neutrino in MeV from info-file (float):
interval_E_neutrino_signal = info_signal[2]
# get minimum of E_neutrino in MeV from info-file (float):
min_E_neutrino_signal = info_signal[0]
# get maximum of E_neutrino in MeV from info-file (float):
max_E_neutrino_signal = info_signal[1]
# get bin-width of E_visible in MeV from info-file (float):
interval_E_visible_signal = info_signal[5]
# get minimum of E_visible in MeV from info-file (float):
min_E_visible_signal = info_signal[3]
# get maximum of E_visible in MeV from info-file (float):
max_E_visible_signal = info_signal[4]
# get exposure time in years from info-file (float):
t_years_signal = info_signal[6]
# get number of free protons in the detector from info-file(float):
N_target_signal = info_signal[7]
# get detection efficiency for Inverse Beta Decay from info-file (float):
det_eff_signal = info_signal[8]
# get DM mass in MeV from info-file (float):
mass_DM = info_signal[9]
# get number of neutrinos from the spectrum from info-file (float):
N_neutrino_signal = info_signal[11]
# get DM annihilation cross-section in cm**3/s from info-file (float):
sigma_annihilation = info_signal[12]
# get angular-averaged intensity over whole Milky Way from info.file (float):
J_avg = info_signal[13]
# get electron-antineutrino flux at Earth in 1/(MeV*s*cm**2) from info-file (float):
flux_signal = info_signal[14]

# Load the calculated spectrum of the DSNB background from the txt-file:
# load calculated spectrum from corresponding file (np.array of float):
Spectrum_DSNB = np.loadtxt(file_DSNB)
# load information about the above file (np.array of float):
info_DSNB = np.loadtxt(file_DSNB_info)
# get bin-width of E_neutrino in MeV from info-file (float):
interval_E_neutrino_DSNB = info_DSNB[2]
# get minimum of E_neutrino in MeV from info-file (float):
min_E_neutrino_DSNB = info_DSNB[0]
# get maximum of E_neutrino in MeV from info-file (float):
max_E_neutrino_DSNB = info_DSNB[1]
# get bin-width of E_visible in MeV from info-file (float):
interval_E_visible_DSNB = info_DSNB[5]
# get minimum of E_visible in MeV from info-file (float):
min_E_visible_DSNB = info_DSNB[3]
# get maximum of E_visible in MeV from info-file (float):
max_E_visible_DSNB = info_DSNB[4]
# get exposure time in years from info-file (float):
t_years_DSNB = info_DSNB[6]
# get number of free protons in the detector from info-file(float):
N_target_DSNB = info_DSNB[7]
# get detection efficiency for Inverse Beta Decay from info-file (float):
det_eff_DSNB = info_DSNB[8]
# get number of neutrinos from the spectrum from info-file (float):
N_neutrino_DSNB = info_DSNB[10]
# get mean energy of nu_Ebar in MeV from info-file (float):
E_mean_NuEbar = info_DSNB[11]
# get pinching factor for nu_Ebar from info-file (float):
beta_NuEbar = info_DSNB[12]
# get mean energy of nu_Xbar in MeV from info-file (float):
E_mean_NuXbar = info_DSNB[13]
# get pinching factor for nu_Xbar from info-file (float):
beta_NuXbar = info_DSNB[14]
# get correction factor of SFR from info-file (float):
f_star = info_DSNB[15]

# load the calculated spectrum of the atmospheric CC background from txt-file:
# load calculated spectrum from corresponding file (np.array of float):
Spectrum_CCatmo = np.loadtxt(file_CCatmo)
# load information about the above file (np.array of float):
info_CCatmo = np.loadtxt(file_CCatmo_info)
# get bin-width of E_neutrino in MeV from info-file (float):
interval_E_neutrino_CCatmo = info_CCatmo[2]
# get minimum of E_neutrino in MeV from info-file (float):
min_E_neutrino_CCatmo = info_CCatmo[0]
# get maximum of E_neutrino in MeV from info-file (float):
max_E_neutrino_CCatmo = info_CCatmo[1]
# get bin-width of E_visible in MeV from info-file (float):
interval_E_visible_CCatmo = info_CCatmo[5]
# get minimum of E_visible in MeV from info-file (float):
min_E_visible_CCatmo = info_CCatmo[3]
# get maximum of E_visible in MeV from info-file (float):
max_E_visible_CCatmo = info_CCatmo[4]
# get exposure time in years from info-file (float):
t_years_CCatmo = info_CCatmo[6]
# get number of free protons in the detector from info-file(float):
N_target_CCatmo = info_CCatmo[7]
# get detection efficiency for Inverse Beta Decay from info-file (float):
det_eff_CCatmo = info_CCatmo[8]
# get number of neutrinos from the spectrum from info-file (float):
N_neutrino_CCatmo = info_CCatmo[10]
# get info if oscillation is considered (0=no, 1=yes) from info-file (float):
Oscillation = info_CCatmo[11]
# get survival probability of nu_Ebar (P_e_to_e) from info-file (float):
P_ee = info_CCatmo[12]
# get oscillation probability of nu_Mubar to nu_Ebar (P_mu_to_e) from info-file (float):
P_mue = info_CCatmo[13]

# load the calculated spectrum of the reactor background from txt-file:
# load calculated spectrum from corresponding file (np.array of float):
Spectrum_reactor = np.loadtxt(file_reactor)
# load information about the above file (np.array of float):
info_reactor = np.loadtxt(file_reactor_info)
# get bin-width of E_neutrino in MeV from info-file (float):
interval_E_neutrino_reactor = info_reactor[2]
# get minimum of E_neutrino in MeV from info-file (float):
min_E_neutrino_reactor = info_reactor[0]
# get maximum of E_neutrino in MeV from info-file (float):
max_E_neutrino_reactor = info_reactor[1]
# get bin-width of E_visible in MeV from info-file (float):
interval_E_visible_reactor = info_reactor[5]
# get minimum of E_visible in MeV from info-file (float):
min_E_visible_reactor = info_reactor[3]
# get maximum of E_visible in MeV from info-file (float):
max_E_visible_reactor = info_reactor[4]
# get exposure time in years from info-file (float):
t_years_reactor = info_reactor[6]
# get number of free protons in the detector from info-file(float):
N_target_reactor = info_reactor[7]
# get detection efficiency for Inverse Beta Decay from info-file (float):
det_eff_reactor = info_reactor[8]
# get number of neutrinos from the spectrum from info-file (float):
N_neutrino_reactor = info_reactor[10]
# get thermal power of NPP in GW from info-file (float):
power_thermal = info_reactor[11]
# get fission fraction of U235 from info-file (float):
fraction_U235 = info_reactor[12]
# get fission fraction of U238 from info-file (float):
fraction_U238 = info_reactor[13]
# get fission fraction of Pu239 from info-file (float):
fraction_Pu239 = info_reactor[14]
# get fission fraction of Pu241 from info-file (float):
fraction_Pu241 = info_reactor[15]
# get distance from reactor to detector in meter from info-file (float):
L_meter = info_reactor[16]

""" Check, if the 'input'-parameter of the files are the same (are the spectra generated with the same properties?): """
# check interval_E_neutrino:
interval_E_neutrino = compare_4fileinputs(interval_E_neutrino_signal, interval_E_neutrino_DSNB,
                                          interval_E_neutrino_CCatmo, interval_E_neutrino_reactor,
                                          'interval_E_neutrino')
# check min_E_neutrino:
min_E_neutrino = compare_4fileinputs(min_E_neutrino_signal, min_E_neutrino_DSNB, min_E_neutrino_CCatmo,
                                     min_E_neutrino_reactor, 'min_E_neutrino')
# check max_E_neutrino:
max_E_neutrino = compare_4fileinputs(max_E_neutrino_signal, max_E_neutrino_DSNB, max_E_neutrino_CCatmo,
                                     max_E_neutrino_reactor, 'max_E_neutrino')
# check interval_E_visible:
interval_E_visible = compare_4fileinputs(interval_E_visible_signal, interval_E_visible_DSNB,
                                         interval_E_visible_CCatmo, interval_E_visible_reactor,
                                         'interval_E_visible')
# check min_E_visible:
min_E_visible = compare_4fileinputs(min_E_visible_signal, min_E_visible_DSNB, min_E_visible_CCatmo,
                                    min_E_visible_reactor, 'min_E_visible')
# check max_E_visible:
max_E_visible = compare_4fileinputs(max_E_visible_signal, max_E_visible_DSNB, max_E_visible_CCatmo,
                                    max_E_visible_reactor, 'max_E_visible')
# check t_years (exposure time in years):
t_years = compare_4fileinputs(t_years_signal, t_years_DSNB, t_years_CCatmo, t_years_reactor, 't_years')
# check N_target (number of target / number of free protons):
N_target = compare_4fileinputs(N_target_signal, N_target_DSNB, N_target_CCatmo, N_target_reactor, 'N_target')
# check det_eff (IBD detection efficiency):
det_eff = compare_4fileinputs(det_eff_signal, det_eff_DSNB, det_eff_CCatmo, det_eff_reactor, 'det_eff')


""" define the visible energy in MeV (arange till max + interval to get an array from min to max) 
    (np.array of float): """
E_visible = np.arange(min_E_visible, max_E_visible + interval_E_visible, interval_E_visible)

""" Total simulated spectrum for JUNO after t_years years and for DM with mass mass_DM in 1/bin (np.array of float): """
Spectrum_signal_per_bin = Spectrum_signal * interval_E_visible
Spectrum_DSNB_per_bin = Spectrum_DSNB * interval_E_visible
Spectrum_CCatmo_per_bin = Spectrum_CCatmo * interval_E_visible
Spectrum_reactor_per_bin = Spectrum_reactor * interval_E_visible
# Number of events per bin (spectrum per bin) of E_visible (np.array of float):
Spectrum_total_per_bin = (Spectrum_signal_per_bin + Spectrum_DSNB_per_bin +
                          Spectrum_CCatmo_per_bin + Spectrum_reactor_per_bin)

""" Generate datasets (virtual experiments) from the total simulated spectrum: """
# iteration of index has to go to number_dataset_stop + 1, because np.arange(start, stop)
# generate values in the half-open interval [start, stop) -> stop would be excluded
for index in np.arange(dataset_start, dataset_stop + 1):
    # spectrum of the dataset
    # generate for each entry in Spectrum_total_per_bin a poisson-distributed integer-value, which describes the
    # number of events a virtual experiment would measure in the corresponding bin (units: number of events per bin)
    # (np.array of integer):
    Spectrum_dataset_per_bin = np.random.poisson(Spectrum_total_per_bin)
    # to save the data set values, SAVE_DATA must be True:
    if SAVE_DATA:
        # save Spectrum_dataset to txt-dataset-file:
        np.savetxt(path_dataset + '/Dataset_{0:d}.txt'.format(index), Spectrum_dataset_per_bin, fmt='%4.5f',
                   header='Spectrum of virtual experiment in number of events per bin'
                          '(Dataset generated with gen_dataset_v1.py, {0}):\n'
                          'Input files:\n{1},\n{2},\n{3},\n{4}:'
                   .format(now, file_signal, file_DSNB, file_CCatmo, file_reactor))

        # build a step-plot of the generated dataset and save the image to png-file:
        # h1 = plt.figure(1)
        # plt.step(E_visible, Spectrum_dataset_per_bin, where='mid')
        # plt.xlabel('Visible energy in MeV')
        # plt.ylabel('Number of events per bin per {0:.0f}yr (bin={1:.2f}MeV)'.format(t_years, interval_E_visible))
        # plt.title('Spectrum, JUNO will measure after {0:.0f} years for DM of mass = {1:.0f} MeV'
        #           .format(t_years, mass_DM))
        # plt.ylim(ymin=0)
        # plt.xticks(np.arange(min_E_visible, max_E_visible, 10))
        # plt.grid()
        # plt.savefig(path_dataset + '/Dataset_{0:d}.png'.format(index))
        # plt.close(h1)

# to save information about the dataset and to save the simulated total spectrum, SAVE_DATA must be True:
if SAVE_DATA:
    # save information about dataset to info-dataset-file:
    np.savetxt(path_dataset + '/info_dataset_{0:d}_to_{1:d}.txt'.format(dataset_start, dataset_stop),
               np.array([interval_E_visible, min_E_visible, max_E_visible, interval_E_neutrino, min_E_neutrino,
                         max_E_neutrino, t_years, N_target, det_eff]),
               fmt='%1.9e',
               header='Information about the dataset-simulation Dataset_{0:d}.txt to Dataset_{1:d}.txt:\n'
                      'Input files:\n{2},\n{3},\n{4},\n{5},\n'
                      'bin-width E_visible in MeV, minimum E_visible in MeV, maximum E_visible in MeV,\n'
                      'bin-width E_neutrino in MeV, minimum E_neutrino in MeV, maximum E_neutrino in MeV\n'
                      'exposure time in years, number of targets (free protons), IBD detection efficiency.'
               .format(dataset_start, dataset_stop, file_signal, file_DSNB, file_CCatmo, file_reactor))
    # save total simulated spectrum to spectrum-file_
    # np.savetxt(path_output + '/spectrum_simulated.txt', Spectrum_total_per_bin, fmt='%4.5f',
    #            header='Total simulated spectrum in events/bin:\n'
    #                   '(sum of single spectra from input files:\n'
    #                   '{0},\n{1},\n{2},\n{3})'.format(file_signal, file_DSNB, file_CCatmo, file_reactor))
