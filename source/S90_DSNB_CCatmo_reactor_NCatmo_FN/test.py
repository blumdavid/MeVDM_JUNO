import numpy as np
from matplotlib import pyplot as plt

path_input = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/"

"""
energy = np.arange(10, 100.5, 0.5)

# spectrum in events/bin:
spectrum_DSNB_new = np.loadtxt(path_input + "S90_DSNB_CCatmo_reactor_NCatmo_FN_newIBD/"
                                            "DSNB_EmeanNuXbar22_bin500keV.txt")
# spectrum in events/MeV:
spectrum_DSNB_old = np.loadtxt(path_input + "S90_DSNB_CCatmo_reactor_NCatmo_FN/DSNB_EmeanNuXbar22_bin500keV.txt")
spectrum_DSNB_old = spectrum_DSNB_old * 0.5

# spectrum in events/bin:
spectrum_reactor_new = np.loadtxt(path_input + "S90_DSNB_CCatmo_reactor_NCatmo_FN_newIBD/"
                                               "Reactor_NH_power36_bin500keV.txt")
# spectrum in events/MeV:
spectrum_reactor_old = np.loadtxt(path_input + "S90_DSNB_CCatmo_reactor_NCatmo_FN/Reactor_NH_power36_bin500keV.txt")
spectrum_reactor_old = spectrum_reactor_old * 0.5

# spectrum in events/bin:
spectrum_CCatmo_new = np.loadtxt(path_input + "S90_DSNB_CCatmo_reactor_NCatmo_FN_newIBD/"
                                              "CCatmo_total_Osc1_bin500keV.txt")
# spectrum in events/MeV
spectrum_CCatmo_old = np.loadtxt(path_input + "S90_DSNB_CCatmo_reactor_NCatmo_FN/CCatmo_total_Osc1_bin500keV.txt")
spectrum_CCatmo_old = spectrum_CCatmo_old * 0.5

h1 = plt.figure(1, figsize=(15, 8))
plt.semilogy(energy, spectrum_DSNB_old, 'r--', drawstyle="steps", label="DSNB old: {0:.2f}"
             .format(np.sum(spectrum_DSNB_old)))
plt.semilogy(energy, spectrum_DSNB_new, 'r-', drawstyle="steps", label="DSNB new: {0:.2f}"
             .format(np.sum(spectrum_DSNB_new)))
plt.semilogy(energy, spectrum_reactor_old, 'b--', drawstyle="steps", label="reactor old: {0:.2f}"
             .format(np.sum(spectrum_reactor_old)))
plt.semilogy(energy, spectrum_reactor_new, 'b-', drawstyle="steps", label="reactor new: {0:.2f}"
             .format(np.sum(spectrum_reactor_new)))
plt.semilogy(energy, spectrum_CCatmo_old, 'g--', drawstyle="steps", label="CCatmo old: {0:.2f}"
             .format(np.sum(spectrum_CCatmo_old)))
plt.semilogy(energy, spectrum_CCatmo_new, 'g-', drawstyle="steps", label="CCatmo new: {0:.2f}"
             .format(np.sum(spectrum_CCatmo_new)))
plt.xlim(min(energy), max(energy))
plt.ylim(ymin=10**(-2), ymax=10**2)
plt.xlabel("Visible energy in MeV")
plt.ylabel("Expected spectrum dN/dE in events/bin (bin-width = 0.5 MeV)")
plt.legend()
plt.grid()
plt.show()
"""

path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN_correctIBDwithCosTheta_onlyDMsignal/"

file_signal_20 = path_spectra + "signal_DMmass20_bin500keV.txt"
file_signal_info_20 = path_spectra + "signal_info_DMmass20_bin500keV.txt"

file_signal_50 = path_spectra + "signal_DMmass50_bin500keV.txt"
file_signal_100 = path_spectra + "signal_DMmass100_bin500keV.txt"

# load the information file:
signal_info = np.loadtxt(file_signal_info_20)
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
# DM annihilation cross-section in cm**3/s:
sigma_Anni = signal_info[12]

signal_20 = np.loadtxt(file_signal_20)
signal_50 = np.loadtxt(file_signal_50)
signal_100 = np.loadtxt(file_signal_100)

N_20 = np.sum(signal_20)
N_50 = np.sum(signal_50)
N_100 = np.sum(signal_100)

h1 = plt.figure(1, figsize=(15, 8))
plt.plot(E_visible, signal_20, 'r-', drawstyle="steps", label='DM mass of 20 MeV (expected events = {0:.2f})'
         .format(N_20))
plt.plot(E_visible, signal_50, 'g-', drawstyle="steps", label='DM mass of 50 MeV (expected events = {0:.2f})'
         .format(N_50))
plt.plot(E_visible, signal_100, 'b-', label='DM mass of 100 MeV (expected events = {0:.2f})'
         .format(N_100), drawstyle="steps")
plt.xlim(E_vis_min, E_vis_max)
plt.ylim(0.0001, 1)
plt.xticks(np.arange(E_vis_min, E_vis_max+5, 5))
plt.xlabel("Visible energy in MeV")
plt.ylabel("Expected spectrum dN/dE in events/bin (bin-width = {0:.2f} MeV)".format(E_vis_bin))
plt.title("Expected energy spectrum of neutrinos from DM self-annihilation in JUNO after {0:.0f} years\n"
          "(for $<\\sigma_Av> = ${1:.1e} $cm^3/s$)"
          .format(t_years, sigma_Anni))

plt.legend()
plt.grid()
plt.show()
