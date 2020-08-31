""" Script to generate the electron-antineutrino visible spectrum for the JUNO detector in the energy range
    from few MeV to hundred MeV (VERSION 2):
    - for signal of DM annihilation for different DM masses
    - for DSNB background for different values of E_mean, beta and f_Star
    - for CC atmospheric neutrino background for different values of Oscillation, Prob_e_to_e, Prob_mu_to_e
    - for reactor anti-neutrino background for different values of power_thermal, Fraction_U235, Fraction_U238,
    Fraction_Pu239, Fraction_Pu241 and L_meter

    Difference of Version 2 to Version 1:
    - the convolution of the theoretical spectrum with the gaussian distribution is calculated with the function
    convolution() from gen_spectrum_functions.py
"""

# import of the necessary packages:
import datetime
import numpy as np
from matplotlib import pyplot
from work.MeVDM_JUNO.source.gen_spectrum_functions import sigma_ibd, darkmatter_signal_v2, dsnb_background_v2, \
    reactor_background_v2, ccatmospheric_background_v3

""" Set boolean values to define, what is simulated in the code, if the data is saved and if spectra are displayed: """
# generate signal from DM annihilation:
DM_SIGNAL = False
# generate DSNB background:
DSNB_BACKGROUND = False
# generate CC atmospheric background:
CCATMOSPHERIC_BACKGROUND = False
# generate reactor antineutrino background:
REACTOR_BACKGROUND = False
# get the NC atmospheric background:
NCATMOSPHERIC_BACKGROUND = False
# generate fast neutron background:
FAST_NEUTRON = True
# save the data:
SAVE_DATA = True
# display the generated spectra:
DISPLAY_SPECTRA = False

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" Dark Matter mass in MeV:"""
# Dark Matter mass in MeV (float):
# TODO-me: change the Dark Matter mass to scan the whole energy range
mass_DM = 20.0

""" energy-array: """
# energy corresponding to the electron-antineutrino energy in MeV (np.array of float64):
interval_E_neutrino = 0.01
E_neutrino = np.arange(10, 115 + interval_E_neutrino, interval_E_neutrino)
# energy corresponding to the visible energy in MeV (np.array of float64):
# TODO-me: what is the best bin size???
interval_E_visible = 0.5
E_visible = np.arange(10, 100 + interval_E_visible, interval_E_visible)

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
# results_16000mm_10MeVto100MeV_500nsto1ms_mult1_2400PEto3400PE_dist500mm_R16000mm_PSD97/)
# (total efficiency of real (reconstructed data):
# TODO-me: muon veto cut is not yet implemented!!!!!!!!
detection_eff = 0.66680

""" Often used values of functions: """
# IBD cross-section for the DM signal in cm**2, must be calculated only for energy = mass_DM (float):
sigma_IBD_signal = sigma_ibd(mass_DM, DELTA, MASS_POSITRON)
# IBD cross-section for the backgrounds in cm**2, must be calculated for the whole energy range E1 (np.array of floats):
sigma_IBD = sigma_ibd(E_neutrino, DELTA, MASS_POSITRON)

""" Consider Efficiency from Pulse Shape Analysis (depending on the PSD efficiency of atmo. NC background): """
# PSD efficiency for real IBD events (see PSD_results.odt in folder /home/astro/blum/juno/atmoNC/data_NC/output_PSD):
# TODO-me: take correct PSD efficiency!!
PSD_eff = 0.0756

# PSD efficiency for fast neutron events (see fast_neutron_summary.ods in folder
# /home/astro/blum/PhD/work/MeVDM_JUNO/fast_neutrons/). Fast neutron efficiency and IBD efficiency depend both on NC
# efficiency.
PSD_eff_fast_neutron = 0.99980

# INFO-me: currently all neutrinos interact in the detector via Inverse Beta Decay. BUT a neutrino can also interact
# INFO-me: via elastic scattering or IBD on bound protons.
# TODO-me: consider also other interactions than the IBD -> NOT 100% of the neutrinos interact via IBD
# INFO-me: -> the rate will decrease!!

# TODO-me: cosmogenic isotopes (Li11, B14), that mimic an IBD signal, must be investigated and added
# INFO-me: both can be reduced by reconstructing the muon (-> dead time of the detector) or by PSD between e- and e+
# INFO-me: Li11 endpoint Q=20.55 MeV
# INFO-me: B14 endpoint Q=20.6 MeV

print("spectrum calculation has started...")

""" SIMULATE SPECTRUM OF THE SIGNAL FROM NEUTRINOS FROM DM ANNIHILATION IN THE MILKY WAY: 
    When DM signal should be simulated, DM_SIGNAL must be True """
if DM_SIGNAL:
    print("... simulation of DM annihilation signal...")

    (Spectrum_signal, N_neutrino_signal_vis, Theo_spectrum_signal, N_neutrino_signal_theo,
     sigma_Anni, J_avg, Flux_signal) = \
        darkmatter_signal_v2(E_neutrino, E_visible, interval_E_visible, mass_DM, sigma_IBD_signal, N_target, time,
                             detection_eff, MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    # consider the PSD efficiency:
    Spectrum_signal_PSD = Spectrum_signal * (1 - PSD_eff)
    N_neutrino_signal_vis_PSD = N_neutrino_signal_vis * (1 - PSD_eff)
    Theo_spectrum_signal_PSD = Theo_spectrum_signal * (1 - PSD_eff)
    N_neutrino_signal_theo_PSD = N_neutrino_signal_theo * (1 - PSD_eff)

    if SAVE_DATA:
        # save Spectrum_signal to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'signal_DMmass{0:.0f}_bin{1:.0f}keV.txt'
                   .format(mass_DM, interval_E_visible*1000), Spectrum_signal, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of DM annihilation signal before PSD '
                          '(calculated with gen_spectrum_v2.py, {0}):'
                          '\nDM mass = {1:.0f} MeV, Theo. number of neutrinos = {2:.2f}, Number of neutrinos from '
                          'spectrum = {3:.2f},'
                          '\nDM annihilation cross-section = {4:1.4e} cm**3/s, binning of '
                          'E_visible = {5:.3f} MeV:'
                   .format(now, mass_DM, N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                           interval_E_visible))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'signal_info_DMmass{0:.0f}_bin{1:.0f}keV.txt'
                   .format(mass_DM, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, mass_DM,
                             N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                             J_avg, Flux_signal]),
                   fmt='%1.9e',
                   header='Information to signal_DMmass{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target,'
                          'IBD detection efficiency,\nDM mass in MeV, '
                          '\ntheo. number of neutrinos, number of neutrinos from spectrum,'
                          '\nDM annihilation cross-section in cm**3/s, '
                          '\nangular-averaged intensity over whole Milky Way, '
                          'nu_e_bar flux at Earth in 1/(MeV*s*cm**2):'
                   .format(mass_DM, interval_E_visible*1000))

        # save Spectrum_signal_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'signal_DMmass{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(mass_DM, interval_E_visible*1000), Spectrum_signal_PSD, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of DM annihilation signal after PSD '
                          '(calculated with gen_spectrum_v2.py, {0}):'
                          '\nDM mass = {1:.0f} MeV, Theo. number of neutrinos = {2:.2f}, Number of neutrinos from '
                          'spectrum = {3:.2f},\nPSD efficiency of IBD events = {6:.5f}, Number of neutrinos from '
                          'spectrum after PSD = {7:.2f},'
                          '\nDM annihilation cross-section = {4:1.4e} cm**3/s, binning of '
                          'E_visible = {5:.3f} MeV:'
                   .format(now, mass_DM, N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                           interval_E_visible, PSD_eff, N_neutrino_signal_vis_PSD))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'signal_info_DMmass{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(mass_DM, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff, mass_DM,
                             N_neutrino_signal_theo, N_neutrino_signal_vis, N_neutrino_signal_vis_PSD, sigma_Anni,
                             J_avg, Flux_signal]),
                   fmt='%1.9e',
                   header='Information to signal_DMmass{0:.0f}_bin{1:.0f}keV_PSD.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target,'
                          'IBD detection efficiency,\nPSD efficiency for IBD events, DM mass in MeV, '
                          '\ntheo. number of neutrinos, number of neutrinos from spectrum,'
                          '\nnumber of neutrinos from spectrum after PSD, DM annihilation cross-section in cm**3/s, '
                          '\nangular-averaged intensity over whole Milky Way, '
                          'nu_e_bar flux at Earth in 1/(MeV*s*cm**2):'
                   .format(mass_DM, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE DSNB ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
    When DSNB background should be simulated, DSNB_BACKGROUND must be True """
if DSNB_BACKGROUND:
    print("... simulation of DSNB background...")

    (Spectrum_DSNB, N_neutrino_DSNB_vis, Theo_spectrum_DSNB, N_neutrino_DSNB_theo,
     E_mean_NuEbar, beta_NuEbar, E_mean_NuXbar, beta_NuXbar, f_Star) = \
        dsnb_background_v2(E_neutrino, E_visible, interval_E_visible, sigma_IBD, N_target, time, detection_eff, C_LIGHT,
                           MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    # consider the PSD efficiency:
    Spectrum_DSNB_PSD = Spectrum_DSNB * (1 - PSD_eff)
    N_neutrino_DSNB_vis_PSD = N_neutrino_DSNB_vis * (1 - PSD_eff)
    Theo_spectrum_DSNB_PSD = Theo_spectrum_DSNB * (1 - PSD_eff)
    N_neutrino_DSNB_theo_PSD = N_neutrino_DSNB_theo * (1 - PSD_eff)

    if SAVE_DATA:
        # save Spectrum_DSNB to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'DSNB_EmeanNuXbar{0:.0f}_bin{1:.0f}keV.txt'
                   .format(E_mean_NuXbar, interval_E_visible*1000), Spectrum_DSNB, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of DSNB background before PSD '
                          '(calculated with gen_spectrum_v2.py, {0}):'
                          '\nTheo. number of neutrinos = {1:.2f}, Number of neutrinos from spectrum = {2:.2f},'
                          '\nmean energy nu_Ebar = {3:.2f} MeV, pinching factor nu_Ebar = {4:.2f}, '
                          '\nmean energy nu_Xbar = {5:.2f} MeV, pinching factor nu_Xbar = {6:.2f}, '
                          'binning of E_visible = {7:.3f} Mev:'
                   .format(now, N_neutrino_DSNB_theo, N_neutrino_DSNB_vis, E_mean_NuEbar, beta_NuEbar,
                           E_mean_NuXbar, beta_NuXbar, interval_E_visible))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'DSNB_info_EmeanNuXbar{0:.0f}_bin{1:.0f}keV.txt'
                   .format(E_mean_NuXbar, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, N_neutrino_DSNB_theo,
                             N_neutrino_DSNB_vis, E_mean_NuEbar, beta_NuEbar, E_mean_NuXbar,
                             beta_NuXbar, f_Star]),
                   fmt='%1.9e',
                   header='Information to simulation DSNB_EmeanNuXbar{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target, '
                          'IBD detection efficiency,\ntheo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum,'
                          '\nmean energy of nu_Ebar in MeV, pinching factor for nu_Ebar, '
                          '\nmean energy of nu_Xbar in MeV, pinching factor for nu_Xbar, \ncorrection factor of SFR:'
                   .format(E_mean_NuXbar, interval_E_visible*1000))

        # save Spectrum_DSNB_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'DSNB_EmeanNuXbar{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(E_mean_NuXbar, interval_E_visible*1000), Spectrum_DSNB_PSD, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of DSNB background after PSD '
                          '(calculated with gen_spectrum_v2.py, {0}):'
                          '\nTheo. number of neutrinos = {1:.2f}, Number of neutrinos from spectrum = {2:.2f},'
                          '\nPSD efficiency of IBD events = {8:.5f}, Number of neutrinos from '
                          'spectrum after PSD = {9:.2f},'
                          '\nmean energy nu_Ebar = {3:.2f} MeV, pinching factor nu_Ebar = {4:.2f}, '
                          '\nmean energy nu_Xbar = {5:.2f} MeV, pinching factor nu_Xbar = {6:.2f}, '
                          'binning of E_visible = {7:.3f} Mev:'
                   .format(now, N_neutrino_DSNB_theo, N_neutrino_DSNB_vis, E_mean_NuEbar, beta_NuEbar,
                           E_mean_NuXbar, beta_NuXbar, interval_E_visible, PSD_eff, N_neutrino_DSNB_vis_PSD))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'DSNB_info_EmeanNuXbar{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(E_mean_NuXbar, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff, N_neutrino_DSNB_theo,
                             N_neutrino_DSNB_vis, N_neutrino_DSNB_vis_PSD, E_mean_NuEbar, beta_NuEbar, E_mean_NuXbar,
                             beta_NuXbar, f_Star]),
                   fmt='%1.9e',
                   header='Information to simulation DSNB_EmeanNuXbar{0:.0f}_bin{1:.0f}keV_PSD.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target, '
                          'IBD detection efficiency,\nPSD efficiency for IBD events, theo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD,'
                          '\nmean energy of nu_Ebar in MeV, pinching factor for nu_Ebar, '
                          '\nmean energy of nu_Xbar in MeV, pinching factor for nu_Xbar, \ncorrection factor of SFR:'
                   .format(E_mean_NuXbar, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE REACTOR ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
    When reactor antineutrino background should be simulated, REACTOR_BACKGROUND must be True """
if REACTOR_BACKGROUND:
    print("... simulation of reactor neutrino background...")

    (Spectrum_reactor, N_neutrino_reactor_vis, Theo_spectrum_reactor, N_neutrino_reactor_theo, power_thermal,
     Fraction_U235, Fraction_U238, Fraction_Pu239, Fraction_Pu241, L_meter) = \
        reactor_background_v2(E_neutrino, E_visible, interval_E_visible, sigma_IBD, N_target, t_years, detection_eff,
                              MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    # consider the PSD efficiency:
    Spectrum_reactor_PSD = Spectrum_reactor * (1 - PSD_eff)
    N_neutrino_reactor_vis_PSD = N_neutrino_reactor_vis * (1 - PSD_eff)
    Theo_spectrum_reactor_PSD = Theo_spectrum_reactor * (1 - PSD_eff)
    N_neutrino_reactor_theo_PSD = N_neutrino_reactor_theo * (1 - PSD_eff)

    if SAVE_DATA:
        # save Spectrum_reactor to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'Reactor_NH_power{0:.0f}_bin{1:.0f}keV.txt'
                   .format(power_thermal, interval_E_visible*1000), Spectrum_reactor, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of reactor background before PSD '
                          '(calculated with gen_spectrum_v2.py, {0}):'
                          '\ntheo. number of neutrinos = {1:.2f}, number of neutrinos from spectrum = {2:.2f},'
                          '\nnormal hierarchy considered, thermal power = {3:.2f} GW, binning of E_visible = {4:.3f} '
                          'MeV:'
                          .format(now, N_neutrino_reactor_theo, N_neutrino_reactor_vis,
                                  power_thermal, interval_E_visible))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'Reactor_info_NH_power{0:.0f}_bin{1:.0f}keV.txt'
                   .format(power_thermal, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, N_neutrino_reactor_theo,
                             N_neutrino_reactor_vis,
                             power_thermal, Fraction_U235, Fraction_U238, Fraction_Pu239,
                             Fraction_Pu241, L_meter]),
                   fmt='%1.9e',
                   header='Information to Reactor_NH_power{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency,'
                          '\ntheo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum,'
                          '\nthermal power in GW, fission fraction of U235, U238, Pu239, Pu241,'
                          '\ndistance reactor to detector in meter:'
                   .format(power_thermal, interval_E_visible*1000))

        # save Spectrum_reactor_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'Reactor_NH_power{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(power_thermal, interval_E_visible*1000), Spectrum_reactor_PSD, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of reactor background after PSD '
                          '(calculated with gen_spectrum_v2.py, {0}):'
                          '\ntheo. number of neutrinos = {1:.2f}, number of neutrinos from spectrum = {2:.2f},'
                          '\nPSD efficiency of IBD events = {5:.5f}, Number of neutrinos from '
                          'spectrum after PSD = {6:.2f},'
                          '\nnormal hierarchy considered, thermal power = {3:.2f} GW, binning of E_visible = {4:.3f} '
                          'MeV:'
                          .format(now, N_neutrino_reactor_theo, N_neutrino_reactor_vis,
                                  power_thermal, interval_E_visible, PSD_eff, N_neutrino_reactor_vis_PSD))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'Reactor_info_NH_power{0:.0f}_bin{1:.0f}keV_PSD.txt'
                   .format(power_thermal, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff, N_neutrino_reactor_theo,
                             N_neutrino_reactor_vis, N_neutrino_reactor_vis_PSD,
                             power_thermal, Fraction_U235, Fraction_U238, Fraction_Pu239,
                             Fraction_Pu241, L_meter]),
                   fmt='%1.9e',
                   header='Information to Reactor_NH_power{0:.0f}_bin{1:.0f}keV_PSD.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency,'
                          '\nPSD efficiency for IBD events, theo. number of neutrinos, '
                          '\nnumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD'
                          '\nthermal power in GW, fission fraction of U235, U238, Pu239, Pu241,'
                          '\ndistance reactor to detector in meter:'
                   .format(power_thermal, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE ATMOSPHERIC CC BACKGROUND IN JUNO:
    When atmospheric cc background should be simulated, CCATMOSPHERIC_BACKGROUND must be True """
if CCATMOSPHERIC_BACKGROUND:
    print("... simulation of atmospheric CC neutrino background...")

    (Spectrum_CCatmospheric, N_neutrino_CCatmospheric_vis, Theo_spectrum_CCatmospheric, N_neutrino_CCatmospheric_theo,
     Oscillation, Prob_e_to_e, Prob_mu_to_e) = \
        ccatmospheric_background_v3(E_neutrino, E_visible, interval_E_visible, sigma_IBD, N_target, time, detection_eff,
                                    MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    # consider the PSD efficiency:
    Spectrum_CCatmospheric_PSD = Spectrum_CCatmospheric * (1 - PSD_eff)
    N_neutrino_CCatmospheric_vis_PSD = N_neutrino_CCatmospheric_vis * (1 - PSD_eff)
    Theo_spectrum_CCatmospheric_PSD = Theo_spectrum_CCatmospheric * (1 - PSD_eff)
    N_neutrino_CCatmospheric_theo_PSD = N_neutrino_CCatmospheric_theo * (1 - PSD_eff)

    if SAVE_DATA:
        # save Spectrum_CCatmospheric to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'CCatmo_Osc{0:d}_bin{1:.0f}keV.txt'
                   .format(Oscillation, interval_E_visible*1000), Spectrum_CCatmospheric, fmt='%1.5e',
                   header='Spectrum in 1/MeV of CC atmospheric electron-antineutrino background before PSD '
                          '(calculated with gen_spectrum_v2.py \n'
                          'and function ccatmospheric_background_v3(), {0}):'
                          'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                          '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},'
                          'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                          '\nsurvival probability of nu_Ebar = {4:.2f}, '
                          'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}:'
                   .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                           Prob_e_to_e, Prob_mu_to_e))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'CCatmo_info_Osc{0:d}_bin{1:.0f}keV.txt'
                   .format(Oscillation, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff,
                             N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis,
                             Oscillation,
                             Prob_e_to_e, Prob_mu_to_e]),
                   fmt='%1.9e',
                   header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV.txt:\n'
                          'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency,'
                          '\ntheo. number of neutrinos, '
                          '\nNumber of neutrinos from spectrum,\n'
                          'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                          'oscillation prob. nu_Mubar to nu_Ebar:'
                   .format(Oscillation, interval_E_visible*1000))

        # save Spectrum_CCatmospheric_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'CCatmo_Osc{0:d}_bin{1:.0f}keV_PSD.txt'
                   .format(Oscillation, interval_E_visible*1000), Spectrum_CCatmospheric_PSD, fmt='%1.5e',
                   header='Spectrum in 1/MeV of CC atmospheric electron-antineutrino background after PSD '
                          '(calculated with gen_spectrum_v2.py \n'
                          'and function ccatmospheric_background_v3(), {0}):'
                          'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                          '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},'
                          '\nPSD efficiency of IBD events = {6:.5f}, Number of neutrinos from '
                          'spectrum after PSD = {7:.2f},'
                          'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                          '\nsurvival probability of nu_Ebar = {4:.2f}, '
                          'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}:'
                   .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                           Prob_e_to_e, Prob_mu_to_e, PSD_eff, N_neutrino_CCatmospheric_vis_PSD))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'CCatmo_info_Osc{0:d}_bin{1:.0f}keV_PSD.txt'
                   .format(Oscillation, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, PSD_eff,
                             N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis,
                             N_neutrino_CCatmospheric_vis_PSD, Oscillation,
                             Prob_e_to_e, Prob_mu_to_e]),
                   fmt='%1.9e',
                   header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV_PSD.txt:\n'
                          'Atmospheric CC electron-antineutrino flux at the site of JUNO!\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency,'
                          '\nPSD efficiency for IBD events, theo. number of neutrinos, '
                          '\nNumber of neutrinos from spectrum, number of neutrinos from spectrum after PSD,\n'
                          'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                          'oscillation prob. nu_Mubar to nu_Ebar:'
                   .format(Oscillation, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE FAST NEUTRON BACKGROUND (IBD-like events from fast neutrons) IN JUNO:
    When fast neutron background should be simulated, FAST_NEUTRON must be True """
if FAST_NEUTRON:
    print("... simulation of fast neutron background...")

    # number of IBD-like fast neutron events after 10 years for R < 16 m in the energy window from 10 MeV to 100 MeV:
    N_fast_neutron = 34.25
    # number of IBD-like fast neutron events after PSD:
    N_fast_neutron_PSD = N_fast_neutron * (1 - PSD_eff_fast_neutron)

    # INFO-me: assume flat fast neutron background for JUNO!
    # calculate number of bins in E_visible:
    number_bins = int((E_visible[-1] - E_visible[0]) / interval_E_visible + 1)
    # calculate the number of IBD-like fast neutron events per bin:
    N_fast_neutron_per_bin = N_fast_neutron / number_bins
    # INFO-me: spectrum in events per bin and NOT events per MeV!!
    # create fast neutron spectrum in events per bin with np.full(length of array, value to fill):
    Spectrum_fast_neutron = np.full(number_bins, N_fast_neutron_per_bin)

    # Consider fast neutron PSD efficiency:
    Spectrum_fast_neutron_PSD = Spectrum_fast_neutron * (1 - PSD_eff_fast_neutron)

    # 'theoretical' spectrum in events/per (same like visible spectrum):
    Theo_spectrum_fast_neutron = Spectrum_fast_neutron
    N_fast_neutron_theo = N_fast_neutron
    # 'theoretical' spectrum after PSD:
    Theo_spectrum_fast_neutron_PSD = Spectrum_fast_neutron_PSD
    N_fast_neutron_theo_PSD = N_fast_neutron_PSD

    if SAVE_DATA:
        # save Spectrum_fast_neutron to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'fast_neutron_{0:.0f}events_bin{1:.0f}keV.txt'
                   .format(N_fast_neutron, interval_E_visible*1000), Spectrum_fast_neutron, fmt='%1.5e',
                   header='Spectrum in events/bin of IBD-like fast neutron background before PSD '
                          '(calculated with gen_spectrum_v2.py \n'
                          'and based on file fast_neutron_summary.ods, {0}):'
                          '\nTheo. number of events = {1:.6f}, Number of events from visible spectrum = {2:.6f},'
                   .format(now, N_fast_neutron_theo, N_fast_neutron))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'fast_neutron_info_{0:.0f}events_bin{1:.0f}keV.txt'
                   .format(N_fast_neutron, interval_E_visible*1000),
                   np.array([E_visible[0], E_visible[-1], interval_E_visible, t_years, 16.0,
                             N_fast_neutron_theo, N_fast_neutron]),
                   fmt='%1.9e',
                   header='Information to fast_neutron_{0:.0f}events_bin{1:.0f}keV.txt:\n'
                          'values below:'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, fiducial volume R in m,'
                          'theo. number of events, '
                          '\nNumber of events from spectrum:'
                   .format(N_fast_neutron, interval_E_visible*1000))

        # save Spectrum_fast_neutron_PSD to txt-spectrum-file and information about simulation in txt-info-file:
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'fast_neutron_{0:.0f}events_bin{1:.0f}keV_PSD.txt'
                   .format(N_fast_neutron, interval_E_visible*1000), Spectrum_fast_neutron_PSD, fmt='%1.5e',
                   header='Spectrum in events/bin of IBD-like fast neutron background after PSD '
                          '(calculated with gen_spectrum_v2.py \n'
                          'and based on file fast_neutron_summary.ods, {0}):'
                          '\nTheo. number of events = {1:.6f}, Number of events from visible spectrum = {2:.6f},'
                          '\nfast neutron PSD efficiency = {3:.5f}, number of events after PSD = {4:.3f}'
                   .format(now, N_fast_neutron_theo, N_fast_neutron, PSD_eff_fast_neutron, N_fast_neutron_PSD))
        np.savetxt('/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/'
                   'fast_neutron_info_{0:.0f}events_bin{1:.0f}keV_PSD.txt'
                   .format(N_fast_neutron, interval_E_visible*1000),
                   np.array([E_visible[0], E_visible[-1], interval_E_visible, t_years, 16.0,
                             N_fast_neutron_theo, N_fast_neutron, PSD_eff_fast_neutron, N_fast_neutron_PSD]),
                   fmt='%1.9e',
                   header='Information to fast_neutron_{0:.0f}events_bin{1:.0f}keV_PSD.txt:\n'
                          'values below:'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, fiducial volume R in m, '
                          'theo. number of events, '
                          '\nNumber of events from spectrum, fast neutron PSD efficiency,\n'
                          'number of events after PSD:'
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
