""" Script to generate the electron-antineutrino visible spectrum for the JUNO detector in the energy range
    from few MeV to hundred MeV (VERSION 1):
    - for signal of DM annihilation for different DM masses
    - for DSNB background for different values of E_mean, beta and f_Star
    - for CC atmospheric neutrino background for different values of Oscillation, Prob_e_to_e, Prob_mu_to_e
    - for reactor anti-neutrino background for different values of power_thermal, Fraction_U235, Fraction_U238,
    Fraction_Pu239, Fraction_Pu241 and L_meter

    - The convolution of the theoretical spectrum with the gaussian distribution (defined by energy resolution
    and correlation between visible and neutrino energy) is not correct!!! It depends strongly on the values of
    E_visible and their binning. (see notes from 23.11.17 to 28.11.17)
    - Different calculations of the convolution and their results can be made with the script test_convolution.py
    """

# import of the necessary packages:
import datetime
import numpy as np
from matplotlib import pyplot
from work.MeVDM_JUNO.source.gen_spectrum_functions import sigma_ibd, darkmatter_signal_v1, dsnb_background_v1, \
    reactor_background_v1, ccatmospheric_background_v1

""" Set boolean values to define, what is simulated in the code, if the data is saved and if spectra are displayed: """
# generate signal from DM annihilation:
DM_SIGNAL = False
# generate DSNB background:
DSNB_BACKGROUND = False
# generate CC atmospheric background:
CCATMOSPHERIC_BACKGROUND = True
# generate reactor antineutrino background:
REACTOR_BACKGROUND = False
# save the data:
SAVE_DATA = True
# display the generated spectra:
DISPLAY_SPECTRA = True

""" Variable, which defines the date and time of running the script: """
# get the date and time, when the script was run:
date = datetime.datetime.now()
now = date.strftime("%Y-%m-%d %H:%M")

""" Dark Matter mass in MeV:"""
# Dark Matter mass in MeV (float):
mass_DM = 20.0

""" energy-array: """
# energy corresponding to the electron-antineutrino energy in MeV (np.array of float64):
interval_E_neutrino = 0.01
E_neutrino = np.arange(10, 115 + interval_E_neutrino, interval_E_neutrino)
# energy corresponding to the visible energy in MeV (np.array of float64):
interval_E_visible = 0.1
E_visible = np.arange(10, 100 + interval_E_visible, interval_E_visible)

""" Natural constants: """
# velocity of light in vacuum, in cm/s (float constant):
C_LIGHT = 2.998 * 10 ** 10
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

""" Constants depending on JUNO: """
# total exposure time in years (float):
t_years = 10
# total time-exposure in seconds, 10 years (float):
time = t_years * 3.156 * 10 ** 7
# Number of free protons (target particles) for IBD in JUNO (float):
N_target = 1.45 * 10 ** 33
# detection efficiency of IBD in JUNO, from physics_report.pdf, page 40, table 2.1
# (combined efficiency of energy cut, time cut, vertex cut, Muon veto, fiducial volume) (float):
detection_eff = 0.73  # TODO: detection efficiency is set to 73 from physics_report p. 40 table 2.1

""" Often used values of functions: """
# IBD cross-section for the DM signal in cm**2, must be calculated only for energy = mass_DM (float):
sigma_IBD_signal = sigma_ibd(mass_DM, DELTA, MASS_POSITRON)
# IBD cross-section for the backgrounds in cm**2, must be calculated for the whole energy range E1 (np.array of floats):
sigma_IBD = sigma_ibd(E_neutrino, DELTA, MASS_POSITRON)


# TODO: Neutral Current atmospheric background has to be added


print("spectrum calculation is started...")

""" SIMULATE SPECTRUM OF THE SIGNAL FROM NEUTRINOS FROM DM ANNIHILATION IN THE MILKY WAY: 
    When DM signal should be simulated, DM_SIGNAL must be True """
if DM_SIGNAL:
    print("... simulation of DM annihilation signal...")

    (Spectrum_signal, N_neutrino_signal_vis, Theo_spectrum_signal, N_neutrino_signal_theo,
     sigma_Anni, J_avg, Flux_signal) = \
        darkmatter_signal_v1(E_neutrino, E_visible, mass_DM, sigma_IBD_signal, N_target, time, detection_eff,
                             MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    if SAVE_DATA:
        # save Spectrum_signal to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('gen_spectrum_v1/signal_DMmass{0:.0f}_bin{1:.0f}keV.txt'
                   .format(mass_DM, interval_E_visible*1000), Spectrum_signal, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of DM annihilation signal '
                          '(calculated with gen_spectrum_v1.py, {0}):'
                          '\nDM mass = {1:.0f} MeV, Theo. number of neutrinos = {2:.5f}, Number of neutrinos from '
                          'spectrum = {3:.5f}, \nDM annihilation cross-section = {4:1.4e} cm**3/s, binning of '
                          'E_visible = {5:.3f} MeV:'
                   .format(now, mass_DM, N_neutrino_signal_theo, N_neutrino_signal_vis, sigma_Anni,
                           interval_E_visible))
        np.savetxt('gen_spectrum_v1/signal_info_DMmass{0:.0f}_bin{1:.0f}keV.txt'
                   .format(mass_DM, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, mass_DM, N_neutrino_signal_theo,
                             N_neutrino_signal_vis, sigma_Anni, J_avg, Flux_signal]),
                   fmt='%1.9e',
                   header='Information to signal_DMmass{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,'
                          '\nexposure time t_years in years, number of free protons N_target,'
                          'IBD detection efficiency, \nDM mass in MeV, theo. number of neutrinos, '
                          'number of neutrinos from spectrum, DM annihilation cross-section in cm**3/s, '
                          '\nangular-averaged intensity over whole Milky Way, '
                          'nu_e_bar flux at Earth in 1/(MeV*s*cm**2):'
                   .format(mass_DM, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE DSNB ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
    When DSNB background should be simulated, DSNB_BACKGROUND must be True """
if DSNB_BACKGROUND:
    print("... simulation of DSNB background...")

    (Spectrum_DSNB, N_neutrino_DSNB_vis, Theo_spectrum_DSNB, N_neutrino_DSNB_theo,
     E_mean_NuEbar, beta_NuEbar, E_mean_NuXbar, beta_NuXbar, f_Star) = \
        dsnb_background_v1(E_neutrino, E_visible, sigma_IBD, N_target, time, detection_eff, C_LIGHT,
                           MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    if SAVE_DATA:
        # save Spectrum_DSNB to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('gen_spectrum_v1/DSNB_EmeanNuXbar{0:.0f}_bin{1:.0f}keV.txt'
                   .format(E_mean_NuXbar, interval_E_visible*1000), Spectrum_DSNB, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of DSNB background '
                          '(calculated with gen_spectrum_v1.py, {0}):'
                          '\nTheo. number of neutrinos = {1:.5f}, Number of neutrinos from spectrum = {2:.5f},'
                          '\nmean energy nu_Ebar = {3:.2f} MeV, pinching factor nu_Ebar = {4:.2f}, '
                          '\nmean energy nu_Xbar = {5:.2f} MeV, pinching factor nu_Xbar = {6:.2f}, '
                          'binning of E_visible = {7:.3f} Mev:'
                   .format(now, N_neutrino_DSNB_theo, N_neutrino_DSNB_vis, E_mean_NuEbar, beta_NuEbar,
                           E_mean_NuXbar, beta_NuXbar, interval_E_visible))
        np.savetxt('gen_spectrum_v1/DSNB_info_EmeanNuXbar{0:.0f}_bin{1:.0f}keV.txt'
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
                          'IBD detection efficiency, \ntheo. number of neutrinos, number of neutrinos from spectrum, '
                          '\nmean energy of nu_Ebar in MeV, pinching factor for nu_Ebar, '
                          '\nmean energy of nu_Xbar in MeV, pinching factor for nu_Xbar, \ncorrection factor of SFR:'
                   .format(E_mean_NuXbar, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE REACTOR ELECTRON-ANTINEUTRINO BACKGROUND IN JUNO: 
    When reactor antineutrino background should be simulated, REACTOR_BACKGROUND must be True """
if REACTOR_BACKGROUND:
    print("... simulation of reactor neutrino background...")

    (Spectrum_reactor, N_neutrino_reactor_vis, Theo_spectrum_reactor, N_neutrino_reactor_theo, power_thermal,
     Fraction_U235, Fraction_U238, Fraction_Pu239, Fraction_Pu241, L_meter) = \
        reactor_background_v1(E_neutrino, E_visible, sigma_IBD, N_target, t_years, detection_eff,
                              MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    if SAVE_DATA:
        # save Spectrum_reactor to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('gen_spectrum_v1/Reactor_NH_power{0:.0f}_bin{1:.0f}keV.txt'
                   .format(power_thermal, interval_E_visible*1000), Spectrum_reactor, fmt='%1.5e',
                   header='Spectrum in electron-antineutrino/MeV of reactor background '
                          '(calculated with gen_spectrum_v1.py, {0}):'
                          '\ntheo. number of neutrinos = {1:.6f}, number of neutrinos from spectrum = {2:.5f},\n'
                          'normal hierarchy considered, thermal power = {3:.2f} GW, binning of E_visible = {4:.3f} MeV:'
                          .format(now, N_neutrino_reactor_theo, N_neutrino_reactor_vis,
                                  power_thermal, interval_E_visible))
        np.savetxt('gen_spectrum_v1/Reactor_info_NH_power{0:.0f}_bin{1:.0f}keV.txt'
                   .format(power_thermal, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff, N_neutrino_reactor_theo,
                             N_neutrino_reactor_vis, power_thermal, Fraction_U235, Fraction_U238, Fraction_Pu239,
                             Fraction_Pu241, L_meter]),
                   fmt='%1.9e',
                   header='Information to Reactor_NH_power{0:.0f}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency, '
                          '\ntheo. number of neutrinos, number of neutrinos from spectrum, '
                          '\nthermal power in GW, fission fraction of U235, U238, Pu239, Pu241,'
                          '\ndistance reactor to detector in meter:'
                   .format(power_thermal, interval_E_visible*1000))

""" SIMULATE SPECTRUM OF THE ATMOSPHERIC CC BACKGROUND IN JUNO:
    When atmospheric cc background should be simulated, CCATMOSPHERIC_BACKGROUND must be True """
if CCATMOSPHERIC_BACKGROUND:
    print("... simulation of atmospheric CC neutrino background...")

    (Spectrum_CCatmospheric, N_neutrino_CCatmospheric_vis, Theo_spectrum_CCatmospheric, N_neutrino_CCatmospheric_theo,
     Oscillation, Prob_e_to_e, Prob_mu_to_e) = \
        ccatmospheric_background_v1(E_neutrino, E_visible, sigma_IBD, N_target, time, detection_eff,
                                    MASS_PROTON, MASS_NEUTRON, MASS_POSITRON)

    if SAVE_DATA:
        # save Spectrum_CCatmospheric to txt-spectrum-file and information about simulation in txt-info-file:
        print("... save data of spectrum to file...")
        np.savetxt('gen_spectrum_v1/CCatmo_Osc{0:d}_bin{1:.0f}keV.txt'
                   .format(Oscillation, interval_E_visible*1000), Spectrum_CCatmospheric, fmt='%1.5e',
                   header='Spectrum in 1/MeV of CC atmospheric electron-antineutrino background '
                          '(calculated with gen_spectrum_v1.py, {0}):'
                          '\nTheo. number of neutrinos = {1:.6f}, Number of neutrinos from spectrum = {2:.6f},\n'
                          'Is oscillation considered (1=yes, 0=no)? {3:d}, '
                          '\nsurvival probability of nu_Ebar = {4:.2f}, '
                          'oscillation prob. nu_Mubar to nu_Ebar = {5:.2f}:'
                   .format(now, N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                           Prob_e_to_e, Prob_mu_to_e))
        np.savetxt('gen_spectrum_v1/CCatmo_info_Osc{0:d}_bin{1:.0f}keV.txt'
                   .format(Oscillation, interval_E_visible*1000),
                   np.array([E_neutrino[0], E_neutrino[-1], interval_E_neutrino, E_visible[0], E_visible[-1],
                             interval_E_visible, t_years, N_target, detection_eff,
                             N_neutrino_CCatmospheric_theo, N_neutrino_CCatmospheric_vis, Oscillation,
                             Prob_e_to_e, Prob_mu_to_e]),
                   fmt='%1.9e',
                   header='Information to CCatmo_Osc{0:d}_bin{1:.0f}keV.txt:\n'
                          'values below: E_neutrino[0] in MeV, E_neutrino[-1] in MeV, interval_E_neutrino in MeV,'
                          '\nE_visible[0] in MeV, E_visible[-1] in MeV, interval_E_visible in MeV,\n'
                          'exposure time t_years in years, number of free protons N_target, IBD detection efficiency, '
                          '\ntheo. number of neutrinos, Number of neutrinos from spectrum, '
                          'Is oscillation considered (1=yes, 0=no)?, \nsurvival probability of nu_Ebar, '
                          'oscillation prob. nu_Mubar to nu_Ebar:'
                   .format(Oscillation, interval_E_visible*1000))


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

    # Display the expected spectra with the settings below:
    h2 = pyplot.figure(2)
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

    pyplot.show()
