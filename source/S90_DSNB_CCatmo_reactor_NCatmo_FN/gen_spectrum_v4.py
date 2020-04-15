""" Script to generate the electron-antineutrino visible spectrum for the JUNO detector in the energy range
    from few MeV to hundred MeV (VERSION 3):
    - for signal of DM annihilation for different DM masses
    - for DSNB background for different values of E_mean, beta and f_Star
    - for CC atmospheric neutrino background for different values of Oscillation, Prob_e_to_e, Prob_mu_to_e
    - for reactor anti-neutrino background for different values of power_thermal, Fraction_U235, Fraction_U238,
    Fraction_Pu239, Fraction_Pu241 and L_meter
    - for fast neutron background estimation

    Difference of Version 4 to version 3:
    - PSD for all IBD signals is done energy-dependent and not energy independent.


    Difference of Version 3 to version 2:
    - for the signal of DM annihilation for different DM masses, also the angular dependency (cos(theta)) of the IBD
    interaction is considered. In version 2, only an average value for cos(theta) is used.
    This will broaden the shape of the DM signal, especially for higher energies.
    - for the backgrounds, the convolution from the neutrino energy to the visible energy is still calculated with
    the average cos(theta), since these are continuous distributions, where the impact of the angular dependence is not
    that important

    Difference of Version 2 to Version 1:
    - the convolution of the theoretical spectrum with the gaussian distribution is calculated with the function
    convolution() from gen_spectrum_functions.py
"""

# import of the necessary packages:
import datetime
import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import sigma_ibd, darkmatter_signal_v3, dsnb_background_v4, \
    reactor_background_v3, ccatmospheric_background_v5, ccatmospheric_background_v6, ibd_kinematics, \
    generate_vis_spectrum, energy_resolution, double_array_entries

""" Set boolean values to define, what is simulated in the code, if the data is saved and if spectra are displayed: """
# generate signal from DM annihilation:
# DM_SIGNAL = True
DM_SIGNAL = False
# generate DSNB background:
# DSNB_BACKGROUND = True
DSNB_BACKGROUND = False
# generate CC atmospheric background:
CCATMOSPHERIC_BACKGROUND = True
CCATMOSPHERIC_BACKGROUND_onlyProton = True
# CCATMOSPHERIC_BACKGROUND = False
# generate reactor antineutrino background:
# REACTOR_BACKGROUND = True
REACTOR_BACKGROUND = False
# generate fast neutron background:
# FAST_NEUTRON = True
FAST_NEUTRON = False
# save the data:
SAVE_DATA = True
# display the generated spectra:
DISPLAY_SPECTRA = False

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" output path, where the results (txt and png files) are saved: """
output_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"

""" Dark Matter mass in MeV:"""
# Dark Matter mass in MeV (float):
# TODO-me: change the Dark Matter mass to scan the whole energy range
mass_DM = 100.0

""" energy-array: """
# energy corresponding to the electron-antineutrino energy in MeV (np.array of float64):
interval_E_neutrino = 0.01
E_neutrino = np.arange(10, 179 + interval_E_neutrino, interval_E_neutrino)
# energy corresponding to the visible energy in MeV (np.array of float64):
# TODO-me: what is the best bin size???
interval_E_visible = 0.5
# energy window corresponding to E_neutrino:
E_visible_whole_range = np.arange(2, 179 + interval_E_visible, interval_E_visible)
# energy window corresponding to observation region (10 MeV to 100 MeV):
E_vis_min = 10.0
E_vis_max = 100.0
E_visible = np.arange(E_vis_min, E_vis_max + interval_E_visible, interval_E_visible)
# calculate the indices of E_visible, wich belong to 10 MeV, 20 MeV, 30 MeV, 40 MeV:
index_10MeV = int((10.0 - min(E_visible)) / interval_E_visible)
index_20MeV = int((20.0 - min(E_visible)) / interval_E_visible)
index_30MeV = int((30.0 - min(E_visible)) / interval_E_visible)
index_40MeV = int((40.0 - min(E_visible)) / interval_E_visible)

""" set parameters to define random numbers"""
# number of randomly generated values for E_nu:
number_random_E_nu = 1000000
# number_random_E_nu = 1000
# set the bin-width of theta between 0 and 180 degree:
theta_interval = 0.1
# set the number of random values of theta that should be generated:
number_of_theta = 1000
# number_of_theta = 10

""" Natural constants: """
# velocity of light in vacuum, in cm/s (reference PDG 2016) (float constant):
C_LIGHT = 2.998 * 10 ** 10
# mass of positron in MeV (reference PDG 2016) (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (reference PDG 2016) (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (reference PDG 2016) (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

""" Constants depending on JUNO: """
# total exposure time in years (float):
t_years = 10
# total time-exposure in seconds, 10 years (float):
time = t_years * 3.156 * 10 ** 7
# Number of free protons (target particles) for IBD in JUNO (page 18 of JUNO DesignReport) (float):
# INFO-me: for a fiducial volume of 20 kton (R=17.7m), you get 1.45 * 10**33 free protons
N_target = 1.45 * 10 ** 33
# detection efficiency of IBD in JUNO, see total_efficiencies_IBD_wo_PSD.txt
# (folder: /home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/
# DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/
# test_10to20_20to30_30to40_40to100_final/)
# (total efficiency of real (reconstructed data):
detection_eff = 0.67005

# TODO-me: cosmogenic isotopes (Li11, B14), that mimic an IBD signal, must be investigated and added
# INFO-me: both can be reduced by reconstructing the muon (-> dead time of the detector) or by PSD between e- and e+
# INFO-me: Li11 endpoint Q=20.55 MeV
# INFO-me: B14 endpoint Q=20.6 MeV
# INFO-me: muon veto cut leads to dead time of detector:
# consider dead time due to muon veto cut (from my notes 'Cosmogenic Background and Muon Veto' (03.10.2019)).
# Exposure ratio (exposure time with muon veto cut / total exposure time):
exposure_ratio_Muon_veto = 0.9717

""" Often used values of functions: """
# IBD cross-section for the DM signal in cm**2, must be calculated only for energy = mass_DM (float):
sigma_IBD_signal = sigma_ibd(mass_DM, DELTA, MASS_POSITRON)
# IBD cross-section for the backgrounds in cm**2, must be calculated for the whole energy range E1 (np.array of floats):
sigma_IBD = sigma_ibd(E_neutrino, DELTA, MASS_POSITRON)

""" Consider Efficiency from Pulse Shape Analysis (depending on the PSD efficiency of atmo. NC background and 
on energy): """
path_PSD_info = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
                "test_10to20_20to30_30to40_40to100_final/"
# PSD suppression for real IBD events (see folder path_PSD_info):
PSD_eff_total = 0.1160
# load simulated IBD spectrum (from 10 to 100 MeV with bin-width 1 MeV):
array_IBD_spectrum = np.loadtxt(path_PSD_info + "IBDspectrum_woPSD_bin1000keV.txt")
# load simulated IBD spectrum after PSD (from 10 to 100 MeV with bin-width 1 MeV):
array_IBD_spectrum_PSD = np.loadtxt(path_PSD_info + "IBDspectrum_wPSD_bin1000keV.txt")

# calculate PSD survival efficiency of IBD events for each energy bin (bin-width 1 MeV):
# INFO-me: survival efficiency and NOT suppression is calculated!
array_eff_IBD_1MeV = array_IBD_spectrum_PSD / array_IBD_spectrum
# INFO-me: last entry of array_IBD_spectrum = 0, therefore is the last entry of array_eff_IBD_1MeV = NaN!
# Therefore replay the last entry of array_eff_IBD_1MeV by the second last value:
array_eff_IBD_1MeV[-1] = array_eff_IBD_1MeV[-2]

# build new array with efficiencies with same bin-width than E_visible, where the entries of array_eff_IBD are
# 'vervielfacht'. (array_eff_IBD_1MeV = [x, y, z, ...], array_eff_IBD = [x, x, y, y, z, z]):
array_eff_IBD = double_array_entries(array_eff_IBD_1MeV, 1.0, interval_E_visible)
# last entry should not be 'vervielfacht':
array_eff_IBD = array_eff_IBD[:-1]

# PSD suppression for fast neutron events (see fast_neutron_summary.ods in folder
# /home/astro/blum/PhD/work/MeVDM_JUNO/fast_neutrons/). Fast neutron efficiency and IBD efficiency depend both on NC
# efficiency and on the energy:
PSD_eff_FN_total = 0.9994
PSD_eff_FN_10_20 = 1.0
PSD_eff_FN_20_30 = 0.9984
PSD_eff_FN_30_40 = 0.9976
PSD_eff_FN_40_100 = 0.9997

# INFO-me: currently all neutrinos interact in the detector via Inverse Beta Decay. BUT a neutrino can also interact
# INFO-me: via elastic scattering or IBD on bound protons.
# TODO-me: consider also other interactions than the IBD -> NOT 100% of the neutrinos interact via IBD
# INFO-me: -> the rate will decrease!!

# TODO-me: implement efficiency due to t_res cut to separate nu_e CC and nu_mu CC event!!

print("spectrum calculation has started...")

""" SIMULATE SPECTRUM OF THE SIGNAL FROM NEUTRINOS FROM DM ANNIHILATION IN THE MILKY WAY: 
    When DM signal should be simulated, DM_SIGNAL must be True """
if DM_SIGNAL:
    print("... simulation of DM annihilation signal...")

    # get the neutrino energy and number of neutrino events for specific DM mass:
    (N_neutrino_signal_theo, E_neutrino_value, sigma_Anni, J_avg, Flux_signal) = \
        darkmatter_signal_v3(mass_DM, sigma_IBD_signal, N_target, time, detection_eff, exposure_ratio_Muon_veto)

    # get the visible spectrum and number of neutrino events from this spectrum for specific DM mass.
    # Here, the IBD kinematics with angular distribution of cos(theta) are considered:
    # INFO-me: theta_interval and number_of_theta are investigated with script test_IBD_kinematics.py
    # set the bin-width of theta between 0 and 180 degree:
    theta_interval_signal = 0.1
    # set the number of random values of theta that should be generated:
    number_of_theta_signal = 100000

    (Spectrum_signal, N_neutrino_signal_vis) = ibd_kinematics(E_neutrino_value, E_visible, N_neutrino_signal_theo,
                                                              MASS_PROTON, MASS_POSITRON, DELTA, theta_interval_signal,
                                                              number_of_theta_signal)

    # consider the PSD efficiency with array_eff_IBD. Spectrum after PSD in events per bin:
    Spectrum_signal_PSD = Spectrum_signal * array_eff_IBD

    # calculate number of events after PSD:
    N_neutrino_signal_vis_PSD = np.sum(Spectrum_signal_PSD)

    plt.figure(1, figsize=(15, 8))
    plt.step(E_visible, Spectrum_signal / interval_E_visible, "r-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_signal_vis, E_vis_min, E_vis_max))
    plt.step(E_visible, Spectrum_signal_PSD / interval_E_visible, "g-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_signal_vis_PSD, E_vis_min, E_vis_max))
    plt.xlabel("energy in MeV")
    plt.ylabel("spectrum in 1/MeV")
    plt.legend()
    plt.grid()
    plt.savefig(output_path + "signal_spectrum_{0:.0f}MeV.png".format(mass_DM))
    plt.close()

    if SAVE_DATA:
        # save Spectrum_signal to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'signal_DMmass{0:.0f}_bin{1:.0f}keV.txt'
                   .format(mass_DM, interval_E_visible*1000), Spectrum_signal, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/bin of DM annihilation signal before PSD '
                          '(calculated with gen_spectrum_v4.py, {0}):'
                          '\nDM mass = {1:.0f} MeV, Theo. number of neutrinos = {2:.2f}, Number of neutrinos from '
                          'spectrum = {3:.2f},'
                          '\nDM annihilation cross-section = {4:1.4e} cm**3/s, binning of '
                          'E_visible = {5:.3f} MeV:'
                   .format(now, mass_DM, N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                           interval_E_visible))
        np.savetxt(output_path + 'signal_info_DMmass{0:.0f}_bin{1:.0f}keV.txt'
                   .format(mass_DM, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, mass_DM,
                             N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                             J_avg, Flux_signal, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to signal_DMmass{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in bin, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target,'
                          'IBD detection efficiency,\nDM mass in MeV, '
                          '\ntheo. number of neutrinos, number of neutrinos from spectrum,'
                          '\nDM annihilation cross-section in cm**3/s, '
                          '\nangular-averaged intensity over whole Milky Way, '
                          'nu_e_bar flux at Earth in 1/(MeV*s*cm**2)'
                          '\nexposure ratio of muon veto cut:'
                   .format(mass_DM, interval_E_visible*1000))

        # save Spectrum_signal_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'signal_DMmass{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(mass_DM, interval_E_visible*1000), Spectrum_signal_PSD, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/bin of DM annihilation signal after PSD '
                          '(calculated with gen_spectrum_v4.py, {0}):'
                          '\nDM mass = {1:.0f} MeV, Theo. number of neutrinos = {2:.2f}, Number of neutrinos from '
                          'spectrum = {3:.2f},\ntotal PSD efficiency of IBD events = {6:.5f}, Number of neutrinos from '
                          'spectrum after PSD = {7:.2f},'
                          '\nDM annihilation cross-section = {4:1.4e} cm**3/s, binning of '
                          'E_visible = {5:.3f} MeV:'
                   .format(now, mass_DM, N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                           interval_E_visible, PSD_eff_total, N_neutrino_signal_vis_PSD))
        np.savetxt(output_path + 'signal_info_DMmass{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(mass_DM, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff_total, mass_DM,
                             N_neutrino_signal_theo, N_neutrino_signal_vis, N_neutrino_signal_vis_PSD, sigma_Anni,
                             J_avg, Flux_signal, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to signal_DMmass{0:.0f}_bin{1:.0f}keV_PSD.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target,'
                          'IBD detection efficiency,\ntotal PSD efficiency for IBD events, DM mass in MeV, '
                          '\ntheo. number of neutrinos, number of neutrinos from spectrum,'
                          '\nnumber of neutrinos from spectrum after PSD, DM annihilation cross-section in cm**3/s, '
                          '\nangular-averaged intensity over whole Milky Way, '
                          'nu_e_bar flux at Earth in 1/(MeV*s*cm**2),'
                          '\nexposure ratio of muon veto cut:'
                   .format(mass_DM, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE DSNB ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
    When DSNB background should be simulated, DSNB_BACKGROUND must be True """
if DSNB_BACKGROUND:
    print("... simulation of DSNB background...")

    # calculate theoretical spectrum (events/MeV) from DSNB data of Julia (dsnb_background_v4):
    Theo_spectrum_DSNB, N_neutrino_DSNB_theo = dsnb_background_v4(E_neutrino, sigma_IBD, N_target, time,
                                                                  detection_eff, exposure_ratio_Muon_veto)

    # get visible spectrum in events/bin (spectrum is normalized to number_random_E_nu):
    # INFO-me: spectrum is not from 10 to 100 MeV, but for the energy range defined by E_visble_whole_range
    Spectrum_DSNB = generate_vis_spectrum(Theo_spectrum_DSNB, N_neutrino_DSNB_theo, E_neutrino, interval_E_neutrino,
                                          number_random_E_nu, E_visible_whole_range, number_of_theta, theta_interval,
                                          MASS_PROTON, MASS_POSITRON, DELTA)

    # normalize Spectrum_DSNB with the number_random_E_nu numbers to get visible spectrum with correct number of events:
    Spectrum_DSNB = Spectrum_DSNB / number_random_E_nu

    print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
          .format(np.sum(Spectrum_DSNB), min(E_visible_whole_range), max(E_visible_whole_range)))

    # consider the number of theoretical events:
    Spectrum_DSNB = Spectrum_DSNB * N_neutrino_DSNB_theo

    print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV, N_neutrino_theo is "
          "considered)".format(np.sum(Spectrum_DSNB), min(E_visible_whole_range), max(E_visible_whole_range)))

    # get bin corresponding to E_vis_min:
    index_E_vis_min = int((E_vis_min - min(E_visible_whole_range)) / interval_E_visible)
    # get bin corresponding to E_vis_max:
    index_E_vis_max = int((E_vis_max - min(E_visible_whole_range)) / interval_E_visible)

    # take visible spectrum only from 10 to 100 MeV:
    Spectrum_DSNB = Spectrum_DSNB[index_E_vis_min:index_E_vis_max + 1]
    # get number of events in visible spectrum from 10 MeV to 100 MeV:
    N_neutrino_DSNB_vis = np.sum(Spectrum_DSNB)

    print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
          .format(np.sum(Spectrum_DSNB), E_vis_min, E_vis_max))

    # consider the PSD efficiency with array_eff_IBD. Spectrum after PSD in events per bin:
    Spectrum_DSNB_PSD = Spectrum_DSNB * array_eff_IBD

    # calculate number of events after PSD:
    N_neutrino_DSNB_vis_PSD = np.sum(Spectrum_DSNB_PSD)

    plt.figure(2, figsize=(15, 8))
    plt.step(E_neutrino, Theo_spectrum_DSNB, "b-",
             label="theo. spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_DSNB_theo, min(E_visible_whole_range), max(E_visible_whole_range)))
    plt.step(E_visible, Spectrum_DSNB / interval_E_visible, "r-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_DSNB_vis, E_vis_min, E_vis_max))
    plt.step(E_visible, Spectrum_DSNB_PSD / interval_E_visible, "g-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_DSNB_vis_PSD, E_vis_min, E_vis_max))
    plt.xlabel("energy in MeV")
    plt.ylabel("spectrum in 1/MeV")
    plt.title("Simulated DSNB spectrum in JUNO after {0:.0f} years of lifetime".format(t_years))
    plt.legend()
    plt.grid()
    plt.savefig(output_path + "DSNB_spectrum.png")
    plt.close()

    if SAVE_DATA:
        # save Spectrum_DSNB to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'DSNB_bin{0:.0f}keV.txt'
                   .format(interval_E_visible*1000), Spectrum_DSNB, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/bin of DSNB background before PSD '
                          '(calculated with gen_spectrum_v4.py, {0}):'
                          '\nTheo. number of neutrinos = {1:.2f}, Number of neutrinos from spectrum = {2:.2f},'
                          'binning of E_visible = {3:.3f} Mev,\n'
                          'DSNB flux is taken from file 1705.02122_DSNB_flux_Figure_8.txt of paper 1705.02122:'
                   .format(now, N_neutrino_DSNB_theo, N_neutrino_DSNB_vis, interval_E_visible))
        np.savetxt(output_path + 'DSNB_info_bin{0:.0f}keV.txt'
                   .format(interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, N_neutrino_DSNB_theo,
                             N_neutrino_DSNB_vis, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to simulation DSNB_bin{0:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target, '
                          'IBD detection efficiency,\ntheo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum,'
                          '\nexposure ratio of muon veto cut,\n'
                          'DSNB flux is taken from file 1705.02122_DSNB_flux_Figure_8.txt of paper 1705.02122:'
                   .format(interval_E_visible*1000))

        # save Spectrum_DSNB_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'DSNB_bin{0:.0f}keV_PSD.txt'
                   .format(interval_E_visible*1000), Spectrum_DSNB_PSD, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/bin of DSNB background after PSD '
                          '(calculated with gen_spectrum_v4.py, {0}):'
                          '\nTheo. number of neutrinos = {1:.2f}, Number of neutrinos from spectrum = {2:.2f},'
                          '\ntotal PSD efficiency of IBD events = {4:.5f}, Number of neutrinos from '
                          '\nspectrum after PSD = {5:.2f}, binning of E_visible = {3:.3f} Mev:'
                   .format(now, N_neutrino_DSNB_theo, N_neutrino_DSNB_vis, interval_E_visible, PSD_eff_total,
                           N_neutrino_DSNB_vis_PSD))
        np.savetxt(output_path + 'DSNB_info_bin{0:.0f}keV_PSD.txt'.format(interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff_total, N_neutrino_DSNB_theo,
                             N_neutrino_DSNB_vis, N_neutrino_DSNB_vis_PSD, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to simulation DSNB_bin{0:.0f}keV_PSD.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target, '
                          'IBD detection efficiency,\ntotal PSD efficiency for IBD events, DM mass in MeV, '
                          'theo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD,'
                          '\nexposure ratio of muon veto cut:'
                   .format(interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE REACTOR ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
    When reactor antineutrino background should be simulated, REACTOR_BACKGROUND must be True """
if REACTOR_BACKGROUND:
    print("... simulation of reactor neutrino background...")

    # calculate theoretical spectrum with reactor_background_v3():
    (Theo_spectrum_reactor, N_neutrino_reactor_theo, power_thermal, Fraction_U235, Fraction_U238, Fraction_Pu239,
     Fraction_Pu241, L_meter) = \
        reactor_background_v3(E_neutrino, sigma_IBD, N_target, time, detection_eff, exposure_ratio_Muon_veto)

    # get visible spectrum in events/bin (spectrum is normalized to number_random_E_nu):
    # INFO-me: spectrum is not from 10 to 100 MeV, but for the energy range defined by E_visble_whole_range
    Spectrum_reactor = generate_vis_spectrum(Theo_spectrum_reactor, N_neutrino_reactor_theo, E_neutrino,
                                             interval_E_neutrino, number_random_E_nu, E_visible_whole_range,
                                             number_of_theta, theta_interval, MASS_PROTON, MASS_POSITRON, DELTA)

    # normalize Spectrum_reactor with the number_random_E_nu numbers to get visible spectrum with correct number of
    # events:
    Spectrum_reactor = Spectrum_reactor / number_random_E_nu

    print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
          .format(np.sum(Spectrum_reactor), min(E_visible_whole_range), max(E_visible_whole_range)))

    # consider the number of theoretical events:
    Spectrum_reactor = Spectrum_reactor * N_neutrino_reactor_theo

    print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV, N_neutrino_theo is "
          "considered)".format(np.sum(Spectrum_reactor), min(E_visible_whole_range), max(E_visible_whole_range)))

    # get bin corresponding to E_vis_min:
    index_E_vis_min = int((E_vis_min - min(E_visible_whole_range)) / interval_E_visible)
    # get bin corresponding to E_vis_max:
    index_E_vis_max = int((E_vis_max - min(E_visible_whole_range)) / interval_E_visible)

    # take visible spectrum only from 10 to 100 MeV:
    Spectrum_reactor = Spectrum_reactor[index_E_vis_min:index_E_vis_max + 1]
    # get number of events in visible spectrum from 10 MeV to 100 MeV:
    N_neutrino_reactor_vis = np.sum(Spectrum_reactor)

    print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
          .format(np.sum(Spectrum_reactor), E_vis_min, E_vis_max))

    # consider the PSD efficiency with array_eff_IBD. Spectrum after PSD in events per bin:
    Spectrum_reactor_PSD = Spectrum_reactor * array_eff_IBD

    # calculate number of events after PSD:
    N_neutrino_reactor_vis_PSD = np.sum(Spectrum_reactor_PSD)

    plt.figure(3, figsize=(15, 8))
    plt.step(E_neutrino, Theo_spectrum_reactor, "b-",
             label="theo. spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_reactor_theo, min(E_visible_whole_range), max(E_visible_whole_range)))
    plt.step(E_visible, Spectrum_reactor / interval_E_visible, "r-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_reactor_vis, E_vis_min, E_vis_max))
    plt.step(E_visible, Spectrum_reactor_PSD / interval_E_visible, "g-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_neutrino_reactor_vis_PSD, E_vis_min, E_vis_max))
    plt.xlabel("energy in MeV")
    plt.ylabel("spectrum in 1/MeV")
    plt.title("Simulated reactor spectrum in JUNO after {0:.0f} years of lifetime".format(t_years))
    plt.legend()
    plt.grid()
    plt.savefig(output_path + "reactor_spectrum.png")
    plt.close()

    if SAVE_DATA:
        # save Spectrum_reactor to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'Reactor_NH_power{0:.0f}_bin{1:.0f}keV.txt'
                   .format(power_thermal, interval_E_visible*1000), Spectrum_reactor, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/bin of reactor background before PSD '
                          '(calculated with gen_spectrum_v4.py, {0}):'
                          '\ntheo. number of neutrinos = {1:.2f}, number of neutrinos from spectrum = {2:.2f},'
                          '\nnormal hierarchy considered, thermal power = {3:.2f} GW, binning of E_visible = {4:.3f} '
                          'MeV:'
                          .format(now, N_neutrino_reactor_theo, N_neutrino_reactor_vis,
                                  power_thermal, interval_E_visible))
        np.savetxt(output_path + 'Reactor_info_NH_power{0:.0f}_bin{1:.0f}keV.txt'
                   .format(power_thermal, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, N_neutrino_reactor_theo,
                             N_neutrino_reactor_vis,
                             power_thermal, Fraction_U235, Fraction_U238, Fraction_Pu239,
                             Fraction_Pu241, L_meter, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to Reactor_NH_power{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency,'
                          '\ntheo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum,'
                          '\nthermal power in GW, fission fraction of U235, U238, Pu239, Pu241,'
                          '\ndistance reactor to detector in meter,'
                          '\nexposure ratio of muon veto cut:'
                   .format(power_thermal, interval_E_visible*1000))

        # save Spectrum_reactor_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'Reactor_NH_power{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(power_thermal, interval_E_visible*1000), Spectrum_reactor_PSD, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/bin of reactor background after PSD '
                          '(calculated with gen_spectrum_v4.py, {0}):'
                          '\ntheo. number of neutrinos = {1:.2f}, number of neutrinos from spectrum = {2:.2f},'
                          '\ntotal PSD efficiency of IBD events = {5:.5f}, Number of neutrinos from '
                          'spectrum after PSD = {6:.2f},'
                          '\nnormal hierarchy considered, thermal power = {3:.2f} GW, binning of E_visible = {4:.3f} '
                          'MeV:'
                          .format(now, N_neutrino_reactor_theo, N_neutrino_reactor_vis,
                                  power_thermal, interval_E_visible, PSD_eff_total, N_neutrino_reactor_vis_PSD))
        np.savetxt(output_path + 'Reactor_info_NH_power{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(power_thermal, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff_total,
                             N_neutrino_reactor_theo, N_neutrino_reactor_vis, N_neutrino_reactor_vis_PSD,
                             power_thermal, Fraction_U235, Fraction_U238, Fraction_Pu239,
                             Fraction_Pu241, L_meter, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to Reactor_NH_power{0:.0f}_bin{1:.0f}keV_PSD.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency,'
                          '\ntotal PSD efficiency for IBD events, theo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD'
                          '\nthermal power in GW, fission fraction of U235, U238, Pu239, Pu241,'
                          '\ndistance reactor to detector in meter,'
                          '\nexposure ratio of muon veto cut:'
                   .format(power_thermal, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE ATMOSPHERIC CC BACKGROUND IN JUNO:
    When atmospheric cc background should be simulated, CCATMOSPHERIC_BACKGROUND must be True """
if CCATMOSPHERIC_BACKGROUND:
    print("... simulation of atmospheric CC neutrino background...")

    # simulate only atmospheric CC background on proton:
    if CCATMOSPHERIC_BACKGROUND_onlyProton:

        (Theo_spectrum_CCatmospheric, N_neutrino_CCatmospheric_theo, Oscillation, Prob_e_to_e, Prob_mu_to_e) = \
            ccatmospheric_background_v5(E_neutrino, sigma_IBD, N_target, time, detection_eff, exposure_ratio_Muon_veto)

        # get visible spectrum in events/bin (spectrum is normalized to number_random_E_nu):
        # INFO-me: spectrum is not from 10 to 100 MeV, but for the energy range defined by E_visble_whole_range
        Spectrum_CCatmospheric = generate_vis_spectrum(Theo_spectrum_CCatmospheric, N_neutrino_CCatmospheric_theo,
                                                       E_neutrino, interval_E_neutrino, number_random_E_nu,
                                                       E_visible_whole_range, number_of_theta, theta_interval,
                                                       MASS_PROTON, MASS_POSITRON, DELTA)

        # normalize Spectrum_CCatmospheric with the number_random_E_nu numbers to get visible spectrum with correct
        # number of events:
        Spectrum_CCatmospheric = Spectrum_CCatmospheric / number_random_E_nu

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
              .format(np.sum(Spectrum_CCatmospheric), min(E_visible_whole_range), max(E_visible_whole_range)))

        # consider the number of theoretical events:
        Spectrum_CCatmospheric = Spectrum_CCatmospheric * N_neutrino_CCatmospheric_theo

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV, N_neutrino_theo is "
              "considered)".format(np.sum(Spectrum_CCatmospheric), min(E_visible_whole_range),
                                   max(E_visible_whole_range)))

        # get bin corresponding to E_vis_min:
        index_E_vis_min = int((E_vis_min - min(E_visible_whole_range)) / interval_E_visible)
        # get bin corresponding to E_vis_max:
        index_E_vis_max = int((E_vis_max - min(E_visible_whole_range)) / interval_E_visible)

        # take visible spectrum only from 10 to 100 MeV:
        Spectrum_CCatmospheric = Spectrum_CCatmospheric[index_E_vis_min:index_E_vis_max + 1]
        # get number of events in visible spectrum from 10 MeV to 100 MeV:
        N_neutrino_CCatmospheric_vis = np.sum(Spectrum_CCatmospheric)

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
              .format(np.sum(Spectrum_CCatmospheric), E_vis_min, E_vis_max))

        # consider the PSD efficiency with array_eff_IBD. Spectrum after PSD in events per bin:
        Spectrum_CCatmospheric_PSD = Spectrum_CCatmospheric * array_eff_IBD

        # calculate number of events after PSD:
        N_neutrino_CCatmospheric_vis_PSD = np.sum(Spectrum_CCatmospheric_PSD)

        plt.figure(4, figsize=(15, 8))
        plt.step(E_neutrino, Theo_spectrum_CCatmospheric, "b-",
                 label="theo. spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
                 .format(N_neutrino_CCatmospheric_theo, min(E_visible_whole_range), max(E_visible_whole_range)))
        plt.step(E_visible, Spectrum_CCatmospheric / interval_E_visible, "r-",
                 label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
                 .format(N_neutrino_CCatmospheric_vis, E_vis_min, E_vis_max))
        plt.step(E_visible, Spectrum_CCatmospheric_PSD / interval_E_visible, "g-",
                 label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
                 .format(N_neutrino_CCatmospheric_vis_PSD, E_vis_min, E_vis_max))
        plt.xlabel("energy in MeV")
        plt.ylabel("spectrum in 1/MeV")
        plt.title("Simulated atmospheric CC spectrum on protons in JUNO after {0:.0f} years of lifetime"
                  .format(t_years))
        plt.legend()
        plt.grid()
        plt.savefig(output_path + "CCatmo_onlyProton_spectrum.png")
        plt.close()
        # plt.show()

        if SAVE_DATA:
            # save Spectrum_CCatmospheric to txt-spectrum-file and information about simulation in txt-info-file:
            print("... save data of spectrum to file...")
            np.savetxt(output_path + 'CCatmo_Osc{0:d}_bin{1:.0f}keV.txt'
                       .format(Oscillation, interval_E_visible * 1000), Spectrum_CCatmospheric, fmt='%1.5e',
                       header='Spectrum in 1/bin of CC atmospheric electron-antineutrino background before PSD '
                              '(calculated with gen_spectrum_v4.py \n'
                              'and function ccatmospheric_background_v5(), CC only on protons, {0}):'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},'
                              'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                              '\nsurvival probability of nu_Ebar = {4:.2f}, '
                              'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}:'
                       .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                               Prob_e_to_e, Prob_mu_to_e))
            np.savetxt(output_path + 'CCatmo_info_Osc{0:d}_bin{1:.0f}keV.txt'
                       .format(Oscillation, interval_E_visible * 1000),
                       np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                                 interval_E_visible, t_years, N_target, detection_eff,
                                 N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis,
                                 Oscillation,
                                 Prob_e_to_e, Prob_mu_to_e, exposure_ratio_Muon_veto]),
                       fmt='%1.9e',
                       header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV.txt:\n'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                              '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                              'exposure time t_years in years, number of free protons N_target, IBD detection '
                              'efficiency,'
                              '\ntheo. number of neutrinos, '
                              '\nNumber of neutrinos from spectrum,\n'
                              'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                              'oscillation prob. nu_Mubar to nu_Ebar,'
                              '\nexposure ratio of muon veto cut:'
                       .format(Oscillation, interval_E_visible * 1000))

            # save Spectrum_CCatmospheric_PSD to txt-spectrum-file and information about simulation in txt-info-file:
            print("... save data of spectrum to file...")
            np.savetxt(output_path + 'CCatmo_Osc{0:d}_bin{1:.0f}keV_PSD.txt'
                       .format(Oscillation, interval_E_visible * 1000), Spectrum_CCatmospheric_PSD, fmt='%1.5e',
                       header='Spectrum in 1/bin of CC atmospheric electron-antineutrino background after PSD '
                              '(calculated with gen_spectrum_v4.py \n'
                              'and function ccatmospheric_background_v5(), CC only on protons, {0}):'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},'
                              '\ntotal PSD efficiency of IBD events = {6:.5f}, Number of neutrinos from '
                              'spectrum after PSD = {7:.2f},'
                              'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                              '\nsurvival probability of nu_Ebar = {4:.2f}, '
                              'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}:'
                       .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                               Prob_e_to_e, Prob_mu_to_e, PSD_eff_total, N_neutrino_CCatmospheric_vis_PSD))
            np.savetxt(output_path + 'CCatmo_info_Osc{0:d}_bin{1:.0f}keV_PSD.txt'
                       .format(Oscillation, interval_E_visible * 1000),
                       np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                                 interval_E_visible, t_years, N_target, detection_eff, PSD_eff_total,
                                 N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis,
                                 N_neutrino_CCatmospheric_vis_PSD, Oscillation,
                                 Prob_e_to_e, Prob_mu_to_e, exposure_ratio_Muon_veto]),
                       fmt='%1.9e',
                       header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV_PSD.txt:\n'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                              '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                              'exposure time t_years in years, number of free protons N_target, IBD detection '
                              'efficiency,'
                              '\ntotal PSD efficiency for IBD events, theo. number of neutrinos, '
                              '\nNumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD,\n'
                              'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                              'oscillation prob. nu_Mubar to nu_Ebar,'
                              '\nexposure ratio of muon veto cut:'
                       .format(Oscillation, interval_E_visible * 1000))

    else:
        # simulate atmospheric CC background on proton:
        # get theoretical spectrum in 1/MeV:
        (Theo_spectrum_CCatmospheric_proton, N_neutrino_CCatmospheric_theo_proton, Oscillation, Prob_e_to_e,
         Prob_mu_to_e) = \
            ccatmospheric_background_v5(E_neutrino, sigma_IBD, N_target, time, detection_eff, exposure_ratio_Muon_veto)

        # get visible spectrum in events/bin (spectrum is normalized to number_random_E_nu):
        Spectrum_CCatmospheric_proton = generate_vis_spectrum(Theo_spectrum_CCatmospheric_proton,
                                                              N_neutrino_CCatmospheric_theo_proton, E_neutrino,
                                                              interval_E_neutrino, number_random_E_nu,
                                                              E_visible_whole_range, number_of_theta, theta_interval,
                                                              MASS_PROTON, MASS_POSITRON, DELTA)

        # normalize Spectrum_CCatmospheric with the number_random_E_nu numbers to get visible spectrum with correct
        # number of events:
        Spectrum_CCatmospheric_proton = Spectrum_CCatmospheric_proton / number_random_E_nu

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
              .format(np.sum(Spectrum_CCatmospheric_proton), min(E_visible_whole_range), max(E_visible_whole_range)))

        # consider the number of theoretical events:
        Spectrum_CCatmospheric_proton = Spectrum_CCatmospheric_proton * N_neutrino_CCatmospheric_theo_proton

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV, N_neutrino_theo is "
              "considered)".format(np.sum(Spectrum_CCatmospheric_proton), min(E_visible_whole_range),
                                   max(E_visible_whole_range)))

        # simulate atmospheric CC background on C12 (theoretical spectrum in 1/MeV):
        (Theo_spectrum_CCatmospheric_C12, N_neutrino_CCatmospheric_theo_C12, Oscillation, Prob_e_to_e,
         Prob_mu_to_e, number_C12) = ccatmospheric_background_v6(E_neutrino, time, detection_eff,
                                                                 exposure_ratio_Muon_veto)

        # generate visible spectrum with theoretical spectrum:
        # calculate theoretical spectrum in 1/bin (not 1/MeV):
        Theo_spectrum_CCatmospheric_C12 = Theo_spectrum_CCatmospheric_C12 * interval_E_neutrino

        print("number of events in theo. spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
              .format(np.sum(Theo_spectrum_CCatmospheric_C12), min(E_neutrino), max(E_neutrino)))

        # normalize spectrum_theo to 1 to get probability function:
        Theo_spectrum_CCatmospheric_C12_norm = Theo_spectrum_CCatmospheric_C12 / np.sum(Theo_spectrum_CCatmospheric_C12)

        print("normalized theo. spectrum = {0:.3f}".format(np.sum(Theo_spectrum_CCatmospheric_C12_norm)))

        # generate number_random_E_nu random values of E_neutrino from theoretical spectrum (array):
        array_e_neutrino = np.random.choice(E_neutrino, p=Theo_spectrum_CCatmospheric_C12_norm, size=number_random_E_nu)

        # preallocate array, where visible energies are stored:
        array_vis_neutrino = []

        # loop over entries in array_e_neutrino:
        for index in range(len(array_e_neutrino)):

            # interaction channel (nu_e_bar + C12 -> positron + neutron + B11;
            # most likely channel of nu_e_bar + C12 -> positron + neutron + ...):
            # positron energy in MeV (very simple kinematics (nu_e_bar + C12 -> positron + neutron + B11;
            # E_e = E_nu -(m_B11 + m_n - m_C12 + m_e)), kinetic energy of neutron is neglected.
            # To get E_vis: E_vis = E_e + mass_positron:
            vis_neutrino = array_e_neutrino[index] - (11.0093 * 931.494 + MASS_NEUTRON - 12.0 * 931.494 + MASS_POSITRON)
            vis_neutrino += MASS_POSITRON

            # check, if vis_neutrino is positive:
            if vis_neutrino <= 0:
                continue

            # get sigma from energy resolution in MeV (array):
            sigma = energy_resolution(vis_neutrino)

            # smear E_visible with sigma (array):
            vis_neutrino = np.random.normal(vis_neutrino, sigma)

            # append vis_neutrino to array_vis_neutrino:
            array_vis_neutrino.append(vis_neutrino)

        # build histogram of visible spectrum in 1/bin from array_vis_neutrino in whole visible energy range:
        Spectrum_CCatmospheric_C12, bin_edges = np.histogram(array_vis_neutrino,
                                                             bins=np.append(E_visible_whole_range,
                                                                            max(E_visible_whole_range) +
                                                                            interval_E_visible))

        # normalize Spectrum_CCatmospheric_C12 with the number_random_E_nu numbers to get visible spectrum with correct
        # number of events:
        Spectrum_CCatmospheric_C12 = Spectrum_CCatmospheric_C12 / number_random_E_nu

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
              .format(np.sum(Spectrum_CCatmospheric_C12), min(E_visible_whole_range), max(E_visible_whole_range)))

        # consider the number of theoretical events:
        Spectrum_CCatmospheric_C12 = Spectrum_CCatmospheric_C12 * N_neutrino_CCatmospheric_theo_C12

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV, N_neutrino_theo is "
              "considered)".format(np.sum(Spectrum_CCatmospheric_C12), min(E_visible_whole_range),
                                   max(E_visible_whole_range)))

        # calculate total atmospheric CC background (from 2 to 200 MeV, in events/bin):
        Spectrum_CCatmospheric = Spectrum_CCatmospheric_proton + Spectrum_CCatmospheric_C12

        # get bin corresponding to E_vis_min:
        index_E_vis_min = int((E_vis_min - min(E_visible_whole_range)) / interval_E_visible)
        # get bin corresponding to E_vis_max:
        index_E_vis_max = int((E_vis_max - min(E_visible_whole_range)) / interval_E_visible)

        # take visible spectrum only from 10 to 100 MeV:
        Spectrum_CCatmospheric = Spectrum_CCatmospheric[index_E_vis_min:index_E_vis_max + 1]
        # get number of events in visible spectrum from 10 MeV to 100 MeV:
        N_neutrino_CCatmospheric_vis = np.sum(Spectrum_CCatmospheric)

        # calculate total number of events from theoretical spectrum:
        N_neutrino_CCatmospheric_theo = N_neutrino_CCatmospheric_theo_proton + N_neutrino_CCatmospheric_theo_C12

        print("number of events in visible spectrum = {0:.3f} (from {1:.3f} MeV to {2:.3f} MeV)"
              .format(np.sum(Spectrum_CCatmospheric), E_vis_min, E_vis_max))

        # consider the PSD efficiency with array_eff_IBD. Spectrum after PSD in events per bin:
        Spectrum_CCatmospheric_PSD = Spectrum_CCatmospheric * array_eff_IBD

        # calculate number of events after PSD:
        N_neutrino_CCatmospheric_vis_PSD = np.sum(Spectrum_CCatmospheric_PSD)

        plt.figure(5, figsize=(15, 8))
        plt.step(E_visible, Spectrum_CCatmospheric / interval_E_visible, "r-",
                 label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
                 .format(N_neutrino_CCatmospheric_vis, E_vis_min, E_vis_max))
        plt.step(E_visible, Spectrum_CCatmospheric_PSD / interval_E_visible, "g-",
                 label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
                 .format(N_neutrino_CCatmospheric_vis_PSD, E_vis_min, E_vis_max))
        plt.xlabel("energy in MeV")
        plt.ylabel("spectrum in 1/MeV")
        plt.title("Simulated atmospheric CC spectrum on protons and C12 in JUNO after {0:.0f} years of lifetime"
                  .format(t_years))
        plt.legend()
        plt.grid()
        plt.savefig(output_path + "CCatmo_PandC12_spectrum.png")
        plt.close()

        if SAVE_DATA:
            # save Spectrum_CCatmospheric to txt-spectrum-file and information about simulation in txt-info-file:
            print("... save data of spectrum to file...")
            np.savetxt(output_path + 'CCatmo_total_Osc{0:d}_bin{1:.0f}keV.txt'
                       .format(Oscillation, interval_E_visible * 1000), Spectrum_CCatmospheric, fmt='%1.5e',
                       header='Spectrum in 1/MeV of CC atmospheric electron-antineutrino background before PSD\n'
                              '(nu_e_bar + proton -> positron + neutron and nu_e_bar + C12 -> positron + neutron + X)\n'
                              '(calculated with gen_spectrum_v4.py \n'
                              'and function ccatmospheric_background_v5() and ccatmospheric_background_v6(), {0}):'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},'
                              'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                              '\nsurvival probability of nu_Ebar = {4:.2f}, '
                              'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f},:'
                       .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                               Prob_e_to_e, Prob_mu_to_e))
            np.savetxt(output_path + 'CCatmo_total_info_Osc{0:d}_bin{1:.0f}keV.txt'
                       .format(Oscillation, interval_E_visible * 1000),
                       np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                                 interval_E_visible, t_years, N_target, detection_eff,
                                 N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis,
                                 Oscillation,
                                 Prob_e_to_e, Prob_mu_to_e, exposure_ratio_Muon_veto, number_C12]),
                       fmt='%1.9e',
                       header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV.txt:\n'
                              '(nu_e_bar + proton -> positron + neutron and nu_e_bar + C12 -> positron + neutron + X)\n'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                              '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                              'exposure time t_years in years, number of free protons N_target, IBD detection '
                              'efficiency,'
                              '\ntheo. number of neutrinos, '
                              '\nNumber of neutrinos from spectrum,\n'
                              'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                              'oscillation prob. nu_Mubar to nu_Ebar,'
                              '\nexposure ratio of muon veto cut'
                              '\nnumber of C12 atoms:'
                       .format(Oscillation, interval_E_visible * 1000))

            # save Spectrum_CCatmospheric_PSD to txt-spectrum-file and information about simulation in txt-info-file:
            print("... save data of spectrum to file...")
            np.savetxt(output_path + 'CCatmo_total_Osc{0:d}_bin{1:.0f}keV_PSD.txt'
                       .format(Oscillation, interval_E_visible * 1000), Spectrum_CCatmospheric_PSD, fmt='%1.5e',
                       header='Spectrum in 1/MeV of CC atmospheric electron-antineutrino background after PSD\n'
                              '(nu_e_bar + proton -> positron + neutron and nu_e_bar + C12 -> positron + neutron + X)\n'
                              '(calculated with gen_spectrum_v4.py \n'
                              'and function ccatmospheric_background_v5() and ccatmopsheric_background_v6(), {0}):'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},'
                              '\ntotal PSD efficiency of IBD events = {6:.5f}, Number of neutrinos from '
                              'spectrum after PSD = {7:.2f},'
                              'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                              '\nsurvival probability of nu_Ebar = {4:.2f}, '
                              'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}:'
                       .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                               Prob_e_to_e, Prob_mu_to_e, PSD_eff_total, N_neutrino_CCatmospheric_vis_PSD))
            np.savetxt(output_path + 'CCatmo_total_info_Osc{0:d}_bin{1:.0f}keV_PSD.txt'
                       .format(Oscillation, interval_E_visible * 1000),
                       np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                                 interval_E_visible, t_years, N_target, detection_eff, PSD_eff_total,
                                 N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis,
                                 N_neutrino_CCatmospheric_vis_PSD, Oscillation,
                                 Prob_e_to_e, Prob_mu_to_e, exposure_ratio_Muon_veto, number_C12]),
                       fmt='%1.9e',
                       header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV_PSD.txt:\n'
                              '(nu_e_bar + proton -> positron + neutron and nu_e_bar + C12 -> positron + neutron + X)\n'
                              'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                              'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                              '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                              'exposure time t_years in years, number of free protons N_target, IBD detection '
                              'efficiency,'
                              '\ntotal PSD efficiency for IBD events, theo. number of neutrinos, '
                              '\nNumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD,\n'
                              'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                              'oscillation prob. nu_Mubar to nu_Ebar,'
                              '\nexposure ratio of muon veto cut,'
                              '\nnumber of C12 atoms:'
                       .format(Oscillation, interval_E_visible * 1000))

""" SIMULATE SPECTRUM OF THE FAST NEUTRON BACKGROUND (IBD-like events from fast neutrons) IN JUNO:
    When fast neutron background should be simulated, FAST_NEUTRON must be True """
if FAST_NEUTRON:
    print("... simulation of fast neutron background...")

    # number of IBD-like fast neutron events after 10 years for R < 16 m in the energy window from 10 MeV to 100 MeV
    # (consider also exposure ratio of muon veto cut):
    N_fast_neutron = 34.25 * exposure_ratio_Muon_veto
    N_fast_neutron_theo = N_fast_neutron

    # INFO-me: assume flat fast neutron background for JUNO!
    # calculate number of bins in E_visible:
    number_bins = int((E_visible[-1] - E_visible[0]) / interval_E_visible + 1)
    # calculate the number of IBD-like fast neutron events per bin:
    N_fast_neutron_per_bin = N_fast_neutron / number_bins
    # INFO-me: spectrum in events per bin and NOT events per MeV!!
    # create fast neutron spectrum in events per bin with np.full(length of array, value to fill):
    Spectrum_fast_neutron = np.full(number_bins, N_fast_neutron_per_bin)

    # consider PSD efficiency of fast neutron background (energy dependent for 10 to 20 MeV, 20 to 30 MeV,
    # 30 to 40 MeV, 40 to 100 MeV):
    spectrum_FN_PSD_10_20 = Spectrum_fast_neutron[index_10MeV:index_20MeV] * (1.0 - PSD_eff_FN_10_20)
    spectrum_FN_PSD_20_30 = Spectrum_fast_neutron[index_20MeV:index_30MeV] * (1.0 - PSD_eff_FN_20_30)
    spectrum_FN_PSD_30_40 = Spectrum_fast_neutron[index_30MeV:index_40MeV] * (1.0 - PSD_eff_FN_30_40)
    spectrum_FN_PSD_40_100 = Spectrum_fast_neutron[index_40MeV:] * (1.0 - PSD_eff_FN_40_100)

    # build fast neutron spectrum after PSD in 1/bin:
    spectrum_FN_PSD = np.concatenate((spectrum_FN_PSD_10_20, spectrum_FN_PSD_20_30, spectrum_FN_PSD_30_40,
                                      spectrum_FN_PSD_40_100), axis=None)

    # number of fast neutron events after PSD:
    N_fast_neutron_PSD = np.sum(spectrum_FN_PSD)

    plt.figure(6, figsize=(15, 8))
    plt.step(E_visible, Spectrum_fast_neutron / interval_E_visible, "r-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_fast_neutron, E_vis_min, E_vis_max))
    plt.step(E_visible, spectrum_FN_PSD / interval_E_visible, "g-",
             label="visible spectrum (N = {0:.2f} from {1:.2f} MeV to {2:.2f} MeV)"
             .format(N_fast_neutron_PSD, E_vis_min, E_vis_max))
    plt.xlabel("energy in MeV")
    plt.ylabel("spectrum in 1/MeV")
    plt.title("Simulated fast neutron background in JUNO after {0:.0f} years of lifetime"
              .format(t_years))
    plt.legend()
    plt.grid()
    plt.savefig(output_path + "fast_neutron_spectrum.png")
    plt.close()

    if SAVE_DATA:
        # save Spectrum_fast_neutron to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt(output_path + 'fast_neutron_{0:.0f}events_bin{1:.0f}keV.txt'
                   .format(N_fast_neutron, interval_E_visible*1000), Spectrum_fast_neutron, fmt='%1.5e',
                   header='Spectrum in events/bin of IBD-like fast neutron background before PSD '
                          '(calculated with gen_spectrum_v4.py \n'
                          'and based on file fast_neutron_summary.ods, {0}):'
                          '\nTheo. number of events = {1:.6f}, Number of events from visible spectrum = {2:.6f},'
                   .format(now, N_fast_neutron_theo, N_fast_neutron))
        np.savetxt(output_path + 'fast_neutron_info_{0:.0f}events_bin{1:.0f}keV.txt'
                   .format(N_fast_neutron, interval_E_visible*1000),
                   np.array([E_visible[0], E_visible[-1], interval_E_visible, t_years, 16.0,
                             N_fast_neutron_theo, N_fast_neutron, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to fast_neutron_{0:.0f}events_bin{1:.0f}keV.txt:\n'
                          'values below:'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, fiducial volume R in m,'
                          'theo. number of events, '
                          '\nNumber of events from spectrum, '
                          '\nexposure ratio of muon veto cut:'
                   .format(N_fast_neutron, interval_E_visible*1000))

        # save Spectrum_fast_neutron_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        np.savetxt(output_path + 'fast_neutron_{0:.0f}events_bin{1:.0f}keV_PSD.txt'
                   .format(N_fast_neutron, interval_E_visible*1000), spectrum_FN_PSD, fmt='%1.5e',
                   header='Spectrum in events/bin of IBD-like fast neutron background after PSD '
                          '(calculated with gen_spectrum_v4.py \n'
                          'and based on file fast_neutron_summary.ods, {0}):'
                          '\nTheo. number of events = {1:.6f}, Number of events from visible spectrum = {2:.6f},'
                          '\ntotal fast neutron PSD efficiency = {3:.5f}, number of events after PSD = {4:.3f}'
                   .format(now, N_fast_neutron_theo, N_fast_neutron, PSD_eff_FN_total, N_fast_neutron_PSD))
        np.savetxt(output_path + 'fast_neutron_info_{0:.0f}events_bin{1:.0f}keV_PSD.txt'
                   .format(N_fast_neutron, interval_E_visible*1000),
                   np.array([E_visible[0], E_visible[-1], interval_E_visible, t_years, 16.0,
                             N_fast_neutron_theo, N_fast_neutron, PSD_eff_FN_total, PSD_eff_FN_10_20, PSD_eff_FN_20_30,
                             PSD_eff_FN_30_40, PSD_eff_FN_40_100, N_fast_neutron_PSD, exposure_ratio_Muon_veto]),
                   fmt='%1.9e',
                   header='Information to fast_neutron_{0:.0f}events_bin{1:.0f}keV_PSD.txt:\n'
                          'values below:'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, fiducial volume R in m, '
                          'theo. number of events, '
                          '\nNumber of events from spectrum, total fast neutron PSD efficiency,\n'
                          'FN PSD suppression 10 to 20 MeV, FN PSD suppression 20 to 30 MeV,\n'
                          'FN PSD suppression 30 to 40 MeV, FN PSD suppression 40 to 100 MeV,\n'
                          'number of events after PSD, '
                          '\nexposure ratio of muon veto cut:'
                   .format(N_fast_neutron, interval_E_visible*1000))

""" Display the calculated spectrum: """
if DISPLAY_SPECTRA:
    print("... display plots...")
    # Display the theoretical spectra with the settings below:
    h1 = pyplot.figure(1)
    if DM_SIGNAL:
        pyplot.step(E_neutrino, Theo_spectrum_signal, 'r-', label='signal from DM annihilation for '
                    '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
    if DSNB_BACKGROUND:
        pyplot.step(E_neutrino, Theo_spectrum_DSNB, 'b-', label='DSNB background', where='mid')
    if REACTOR_BACKGROUND:
        pyplot.step(E_neutrino, Theo_spectrum_reactor, 'c-', label='reactor background', where='mid')
    if CCATMOSPHERIC_BACKGROUND:
        pyplot.step(E_neutrino, Theo_spectrum_CCatmospheric, 'g-', label='atmospheric CC background', where='mid')

    pyplot.xlim(E_neutrino[0], E_neutrino[-1])
    pyplot.ylim(ymin=0)
    pyplot.xlabel("Electron-antineutrino energy in MeV")
    pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
    pyplot.title(
        "Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
        .format(t_years, mass_DM))
    pyplot.legend()

    # Display the theoretical spectra after PSD with the settings below:
    h2 = pyplot.figure(2)
    if DM_SIGNAL:
        pyplot.step(E_neutrino, Theo_spectrum_signal_PSD, 'r-', label='signal from DM annihilation for '
                    '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
    if DSNB_BACKGROUND:
        pyplot.step(E_neutrino, Theo_spectrum_DSNB_PSD, 'b-', label='DSNB background', where='mid')
    if REACTOR_BACKGROUND:
        pyplot.step(E_neutrino, Theo_spectrum_reactor_PSD, 'c-', label='reactor background', where='mid')
    if CCATMOSPHERIC_BACKGROUND:
        pyplot.step(E_neutrino, Theo_spectrum_CCatmospheric_PSD, 'g-', label='atmospheric CC background', where='mid')

    pyplot.xlim(E_neutrino[0], E_neutrino[-1])
    pyplot.ylim(ymin=0)
    pyplot.xlabel("Electron-antineutrino energy in MeV")
    pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
    pyplot.title(
        "Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV\n"
        "(PSD efficiency = {2:.3f} %)"
        .format(t_years, mass_DM, PSD_eff*100.0))
    pyplot.legend()


    # Display the expected spectra with the settings below:
    h3 = pyplot.figure(3)
    if DM_SIGNAL:
        pyplot.step(E_visible, Spectrum_signal, 'r-', label='signal from DM annihilation for '
                    '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
    if DSNB_BACKGROUND:
        pyplot.step(E_visible, Spectrum_DSNB, 'b-', label='DSNB background', where='mid')
    if REACTOR_BACKGROUND:
        pyplot.step(E_visible, Spectrum_reactor, 'c-', label='reactor background', where='mid')
    if CCATMOSPHERIC_BACKGROUND:
        pyplot.step(E_visible, Spectrum_CCatmospheric, 'g-', label='atmospheric CC background', where='mid')

    pyplot.xlim(E_visible[0], E_visible[-1])
    pyplot.ylim(ymin=0)
    pyplot.xlabel("Visible energy in MeV")
    pyplot.ylabel("Expected spectrum dN/dE in 1/MeV")
    pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
                 .format(t_years, mass_DM))
    pyplot.legend()

    # Display the expected spectra with the settings below:
    h4 = pyplot.figure(4)
    if DM_SIGNAL:
        pyplot.step(E_visible, Spectrum_signal_PSD, 'r-', label='signal from DM annihilation for '
                    '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
    if DSNB_BACKGROUND:
        pyplot.step(E_visible, Spectrum_DSNB_PSD, 'b-', label='DSNB background', where='mid')
    if REACTOR_BACKGROUND:
        pyplot.step(E_visible, Spectrum_reactor_PSD, 'c-', label='reactor background', where='mid')
    if CCATMOSPHERIC_BACKGROUND:
        pyplot.step(E_visible, Spectrum_CCatmospheric_PSD, 'g-', label='atmospheric CC background', where='mid')

    pyplot.xlim(E_visible[0], E_visible[-1])
    pyplot.ylim(ymin=0)
    pyplot.xlabel("Visible energy in MeV")
    pyplot.ylabel("Expected spectrum dN/dE in 1/MeV")
    pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV\n"
                 "(PSD efficiency = {2:.3f} %)"
                 .format(t_years, mass_DM, PSD_eff*100.0))
    pyplot.legend()


    pyplot.show()
