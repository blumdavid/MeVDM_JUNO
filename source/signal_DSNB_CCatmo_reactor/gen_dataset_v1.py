""" Script to generate several data-sets (virtual experiments) from the spectra,
    which are calculated with gen_spectrum_v2.py (VERSION 1):
"""

# import of the necessary packages:
import datetime
import numpy as np
from work.MeVDM_JUNO.source.gen_spectrum_functions import compare_4fileinputs
from matplotlib import pyplot

# TODO-me: also update the file: gen_dataset_v1_server.py

""" Define parameters for saving of the virtual experiments: 
    number_dataset (dataset_stop - dataset_start) defines, how many datasets (virtual experiments)
    are generated. """
# INFO-me: 10000 datasets might be a good number for the analysis
# dataset_start defines the start point (integer):
dataset_start = 1
# dataset_stop defines the end point (integer):
dataset_stop = 1

""" Set boolean values to define, if the datasets are saved and if spectra are displayed: """
# save the dataset:
SAVE_DATA = True
# display the simulated spectra (when True, you get a warning (can't invoke "event" command: application has been
# destroyed while executing "event generate $w <<ThemeChanged>>" (procedure "ttk::ThemeChanged" line 6) invoked from
# within "ttk::ThemeChanged") -> there is a conflict between the pyplot.close() command and the pyplot.show() command.
# BUT: datasets are generated correctly!!!
DISPLAY_SPECTRA = False

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_10"

""" set the path of the folder, where the datasets are saved: """
path_dataset = path_output + "/datasets"

""" Files, that are read in to generate data-sets (virtual experiments): """
path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2"
file_signal = path_simu + "/signal_DMmass90_bin100keV.txt"
file_signal_info = path_simu + "/signal_info_DMmass90_bin100keV.txt"
file_DSNB = path_simu + "/DSNB_EmeanNuXbar22_bin100keV.txt"
file_DSNB_info = path_simu + "/DSNB_info_EmeanNuXbar22_bin100keV.txt"
file_CCatmo = path_simu + "/CCatmo_Osc1_bin100keV.txt"
file_CCatmo_info = path_simu + "/CCatmo_info_Osc1_bin100keV.txt"
file_reactor = path_simu + "/Reactor_NH_power36_bin100keV.txt"
file_reactor_info = path_simu + "/Reactor_info_NH_power36_bin100keV.txt"

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
    print('Simulate Dataset {0:.0f}'.format(index))
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
        h1 = pyplot.figure(1)
        pyplot.step(E_visible, Spectrum_dataset_per_bin, where='mid')
        pyplot.xlabel('Visible energy in MeV')
        pyplot.ylabel('Number of events per bin per {0:.0f}yr (bin={1:.2f}MeV)'.format(t_years, interval_E_visible))
        pyplot.title('Spectrum, JUNO will measure after {0:.0f} years for DM of mass = {1:.0f} MeV'
                     .format(t_years, mass_DM))
        pyplot.ylim(ymin=0)
        pyplot.xticks(np.arange(min_E_visible, max_E_visible, 10))
        pyplot.grid()
        pyplot.savefig(path_dataset + '/Dataset_{0:d}.png'.format(index))
        pyplot.close(h1)

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
    np.savetxt(path_output + '/spectrum_simulated.txt', Spectrum_total_per_bin, fmt='%4.5f',
               header='Total simulated spectrum in events/bin:\n'
                      '(sum of single spectra from input files:\n'
                      '{0},\n{1},\n{2},\n{3})'.format(file_signal, file_DSNB, file_CCatmo, file_reactor))

# Total simulated spectrum and one dataset is displayed, when DISPLAY_SPECTRA is True:
if DISPLAY_SPECTRA:
    # Display the total simulated spectrum and with different backgrounds:
    h2 = pyplot.figure(2)
    pyplot.step(E_visible, Spectrum_total_per_bin, where='mid', label='total spectrum')
    pyplot.step(E_visible, Spectrum_signal_per_bin, '--', where='mid', label='DM signal ($<\sigma_Av>=${0:.1e}$cm^3/s$)'
                .format(sigma_annihilation))
    pyplot.step(E_visible, Spectrum_DSNB_per_bin, '--', where='mid', label='DSNB background')
    pyplot.step(E_visible, Spectrum_CCatmo_per_bin, '--', where='mid', label='atmospheric CC background')
    pyplot.step(E_visible, Spectrum_reactor_per_bin, '--', where='mid', label='reactor background')
    pyplot.xlabel('Visible energy in MeV')
    pyplot.ylabel('Simulated spectrum in events/(bin*{0:.0f}yr) (bin={1:.2f}MeV)'.format(t_years, interval_E_visible))
    pyplot.title('Simulated spectrum in JUNO after {0:.0f} years for DM of mass = {1:.0f} MeV'.format(t_years, mass_DM))
    pyplot.ylim(ymin=0)
    pyplot.xticks(np.arange(min_E_visible, max_E_visible, 10))
    pyplot.legend()
    pyplot.grid()

    # Display the total simulated spectrum and with different backgrounds (in LOGARITHMIC scale):
    h3 = pyplot.figure(3)
    pyplot.semilogy(E_visible, Spectrum_total_per_bin, label='total spectrum', drawstyle='steps-mid')
    pyplot.semilogy(E_visible, Spectrum_signal_per_bin, '--', label='DM signal ($<\sigma_Av>=${0:.1e}$cm^3/s$)'
                    .format(sigma_annihilation), drawstyle='steps-mid')
    pyplot.semilogy(E_visible, Spectrum_DSNB_per_bin, '--', label='DSNB background', drawstyle='steps-mid')
    pyplot.semilogy(E_visible, Spectrum_CCatmo_per_bin, '--', label='atmospheric CC background', drawstyle='steps-mid')
    pyplot.semilogy(E_visible, Spectrum_reactor_per_bin, '--', label='reactor background', drawstyle='steps-mid')
    pyplot.xlabel('Visible energy in MeV')
    pyplot.ylabel('Simulated spectrum in events/(bin*{0:.0f}yr) (bin={1:.2f}MeV)'.format(t_years, interval_E_visible))
    pyplot.title('Simulated spectrum in JUNO after {0:.0f} years for DM of mass = {1:.0f} MeV'.format(t_years, mass_DM))
    pyplot.ylim(ymin=0.01, ymax=10)
    pyplot.xticks(np.arange(min_E_visible, max_E_visible, 10))
    pyplot.legend()
    pyplot.grid()

    pyplot.show()
