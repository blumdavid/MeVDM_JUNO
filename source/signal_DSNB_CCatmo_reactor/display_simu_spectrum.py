""" Script to display the simulated spectrum (signal spectrum, DSNB background, CCatmo background, Reactor background)
"""
import numpy as np
from matplotlib import pyplot as plt

# path to the directory, where the files of the simulated spectra are saved:
path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/"

# set the file names, where the simulated spectra are saved:
# file_signal = path_spectra + "0signal_bin100keV.txt"
# file_signal_info = path_spectra + "0signal_info_bin100keV.txt"
file_signal = path_spectra + "signal_DMmass20_bin100keV.txt"
file_signal_info = path_spectra + "signal_info_DMmass20_bin100keV.txt"
file_signal_2 = path_spectra + "signal_DMmass100_bin100keV.txt"
file_signal_info_2 = path_spectra + "signal_info_DMmass100_bin100keV.txt"
file_DSNB = path_spectra + "DSNB_EmeanNuXbar22_bin100keV.txt"
file_DSNB_info = path_spectra + "DSNB_info_EmeanNuXbar22_bin100keV.txt"
file_CCatmo = path_spectra + "CCatmo_Osc1_bin100keV.txt"
file_CCatmo_info = path_spectra + "CCatmo_info_Osc1_bin100keV.txt"
file_reactor = path_spectra + "Reactor_NH_power36_bin100keV.txt"
file_reactor_info = path_spectra + "Reactor_info_NH_power36_bin100keV.txt"

# load the spectra (in electron-antineutrinos/MeV) (np.array of float):
signal_per_MeV = np.loadtxt(file_signal)
signal_per_MeV_2 = np.loadtxt(file_signal_2)
DSNB_per_MeV = np.loadtxt(file_DSNB)
CCatmo_per_MeV = np.loadtxt(file_CCatmo)
reactor_per_MeV = np.loadtxt(file_reactor)
# total spectrum in electron-antineutrinos/MeV)
spectrum_per_MeV = signal_per_MeV + DSNB_per_MeV + CCatmo_per_MeV + reactor_per_MeV

# load the information file:
signal_info = np.loadtxt(file_signal_info)
# bin width in MeV:
E_vis_bin = signal_info[5]
# minimal E_vis in MeV:
E_vis_min = signal_info[3]
# maximal E_vis in MeV:
E_vis_max = signal_info[4]
# visible energy in MeV (array of float):
E_visible = np.arange(E_vis_min, E_vis_max+E_vis_bin, E_vis_bin)
# exposure time in years:
t_years = signal_info[6]
# expected number of signal events (float):
Signal_expected_events = signal_info[11]
# DM annihilation cross-section in cm**3/s:
sigma_Anni = signal_info[12]

Signal_expected_events_2 = np.loadtxt(file_signal_info_2)[11]

# load information files of the backgrounds and get the expected number of events from the files:
DSNB_info = np.loadtxt(file_DSNB_info)
DSNB_expected_events = DSNB_info[10]
CCatmo_info = np.loadtxt(file_CCatmo_info)
CCatmo_expected_events = CCatmo_info[10]
Reactor_info = np.loadtxt(file_reactor_info)
Reactor_expected_events = Reactor_info[10]

# spectra in electron-neutrinos/bin) (np.array of float):
signal_per_bin = signal_per_MeV * E_vis_bin
signal_per_bin_2 = signal_per_MeV_2 * E_vis_bin
DSNB_per_bin = DSNB_per_MeV * E_vis_bin
CCatmo_per_bin = CCatmo_per_MeV * E_vis_bin
reactor_per_bin = reactor_per_MeV * E_vis_bin
# total spectrum in electron-antineutrinos/bin:
spectrum_per_bin = signal_per_bin + DSNB_per_bin + CCatmo_per_bin + reactor_per_bin


# Display the expected spectra with the settings below:
h1 = plt.figure(1, figsize=(15, 8))
# plt.step(E_visible, signal_per_MeV, 'r--', label='signal from DM annihilation for '
#          '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
plt.step(E_visible, DSNB_per_MeV, 'b--', label='DSNB background (expected events = {0:.2f})'
         .format(DSNB_expected_events), where='mid')
plt.step(E_visible, reactor_per_MeV, 'c--', label='reactor background (expected events = {0:.2f})'
         .format(Reactor_expected_events), where='mid')
plt.step(E_visible, CCatmo_per_MeV, 'g--', label='atmospheric CC background (expected events = {0:.2f})'
         .format(CCatmo_expected_events), where='mid')
plt.step(E_visible, spectrum_per_MeV, 'k-', label='total spectrum', where='mid')
plt.xlim(E_vis_min, E_vis_max)
plt.ylim(ymin=0)
plt.xlabel("Visible energy in MeV")
plt.ylabel("Expected spectrum dN/dE in 1/MeV")
plt.title("Expected spectrum in JUNO after {0:.0f} years for NO Dark Matter signal".format(t_years))
plt.legend()
plt.grid()

h2 = plt.figure(2, figsize=(15, 8))
# plt.step(E_visible, signal_per_bin, 'r--', label='signal from DM annihilation for '
#          '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
plt.step(E_visible, DSNB_per_bin, 'b--', label='DSNB background (expected events = {0:.2f})'
         .format(DSNB_expected_events), where='mid')
plt.step(E_visible, reactor_per_bin, 'c--', label='reactor background (expected events = {0:.2f})'
         .format(Reactor_expected_events), where='mid')
plt.step(E_visible, CCatmo_per_bin, 'g--', label='atmospheric CC background (expected events = {0:.2f})'
         .format(CCatmo_expected_events), where='mid')
plt.step(E_visible, spectrum_per_bin, 'k-', label='total spectrum', where='mid')
plt.xlim(E_vis_min, E_vis_max)
plt.ylim(ymin=0)
plt.xlabel("Visible energy in MeV")
plt.ylabel("Expected spectrum dN/dE in 1/bin (bin-width = 0.1MeV)")
plt.title("Expected spectrum in JUNO after {0:.0f} years for NO Dark Matter signal".format(t_years))
plt.legend()
plt.grid()

h3 = plt.figure(3, figsize=(15, 8))
# plt.semilogy(E_visible, signal_per_MeV, 'r--', label='signal from DM annihilation for '
#              '$<\sigma_Av>=${0:.1e}$cm^3/s$'.format(sigma_Anni), where='mid')
plt.semilogy(E_visible, DSNB_per_MeV, 'b--', label='DSNB background (expected events = {0:.2f})'
             .format(DSNB_expected_events))
plt.semilogy(E_visible, reactor_per_MeV, 'c--', label='reactor background (expected events = {0:.2f})'
             .format(Reactor_expected_events))
plt.semilogy(E_visible, CCatmo_per_MeV, 'g--', label='atmospheric CC background (expected events = {0:.2f})'
             .format(CCatmo_expected_events))
plt.semilogy(E_visible, spectrum_per_MeV, 'k-', label='total spectrum')
plt.xlim(E_vis_min, E_vis_max)
plt.ylim(ymin=0.1, ymax=100)
plt.xlabel("Visible energy in MeV")
plt.ylabel("Expected spectrum dN/dE in 1/MeV")
plt.title("Expected spectrum in JUNO after {0:.0f} years for NO Dark Matter signal".format(t_years))
plt.legend()
plt.grid()

h4 = plt.figure(4, figsize=(15, 8))
plt.semilogy(E_visible, signal_per_bin, 'r--', label='signal of DM annihilation for DM mass of 20 MeV for '
             '$<\sigma_Av>=${0:.1e}$\,cm^3/s$ (expected events = {1:.2f})'.format(sigma_Anni, Signal_expected_events))
# plt.semilogy(E_visible, signal_per_bin_2, 'm--', label='signal of DM annihilation for DM mass of 100 MeV for '
#              '$<\sigma_Av>=${0:.1e}$\,cm^3/s$ (expected events = {1:.2f})'
#              .format(sigma_Anni, Signal_expected_events_2))
plt.semilogy(E_visible, DSNB_per_bin, 'b--', label='DSNB background (expected events = {0:.2f})'
             .format(DSNB_expected_events))
plt.semilogy(E_visible, reactor_per_bin, 'c--', label='reactor background (expected events = {0:.2f})'
             .format(Reactor_expected_events))
plt.semilogy(E_visible, CCatmo_per_bin, 'g--', label='atmospheric CC background (expected events = {0:.2f})'
             .format(CCatmo_expected_events))
plt.semilogy(E_visible, spectrum_per_bin, 'k-', label='total spectrum')
plt.xlim(E_vis_min, E_vis_max)
plt.ylim(ymin=0.01, ymax=10)
plt.xlabel("Visible energy in MeV")
plt.ylabel("Expected spectrum dN/dE in 1/bin (bin-width = 0.1MeV)")
plt.title("Expected spectrum in JUNO after {0:.0f} years with signal of DM annihilation".format(t_years))
plt.legend()
plt.grid()

plt.show()
