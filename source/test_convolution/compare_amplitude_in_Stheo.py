""" Script to compare the to visible spectra 'signal_DMmass30_100keV_140events.txt' and 'signal_DMmass30_100keV.txt':

    Both spectra are generated with python script gen_spectrum_v2.py, but for file '..._140events.txt' the number of
    neutrinos in the theoretical spectrum was increased to 140 by hand.
    All other properties during spectra calculation (E_neutrino, interval_E_neutrino, E_visible, interval_E_visible,
    exposure time in years, number of free protons, IBD detection efficiency, DM mass in MeV) are the same
    (see 'signal_info_DMmass30_100keV_140events.txt' and 'signal_info_DMmass30_100keV.txt').

    This script is used to check if the shape of the theoretical signal spectrum depend on the amplitude (e.g. number
    of neutrinos).

"""
import numpy as np
from matplotlib import pyplot as plt

""" Load the spectrum files: """
# load 'signal_DMmass30_100keV_140events.txt' (np.array of float):
S_vis_140events = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/"
                             "signal_DMmass30_bin100keV_140events.txt")
# load 'signal_DMmass30_100keV.txt' (np.array of float):
S_vis_1event = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/signal_DMmass30_bin100keV.txt")

""" Load the info files: """
# load 'signal_info_DMmass30_100keV_140events.txt' (np.array of float):
info_140events = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/"
                            "signal_info_DMmass30_bin100keV_140events.txt")
# load 'signal_info_DMmass30_100keV.txt' (np.array of float):
info_1event = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/signal_info_DMmass30_bin100keV.txt")
# get the visible energy in MeV (np.array of float):
E_vis = np.arange(info_1event[3], info_1event[4] + info_1event[5], info_1event[5])
# get the number of events in the visible spectrum (is equal to the number of events in the theo. spectrum) (float):
N_140events = info_140events[11]
# get the number of events in the visible spectrum (is equal to the number of events in the theo. spectrum) (float):
N_1event = info_1event[11]

""" normalize the spectra to 1: """
S_vis_140events_norm = S_vis_140events / np.trapz(S_vis_140events, x=E_vis)
S_vis_1event_norm = S_vis_1event / np.trapz(S_vis_1event, x=E_vis)

""" subtract the normalized spectra from each other: """
S_difference = S_vis_140events_norm - S_vis_1event_norm
# save the array to a txt file:
np.savetxt("/home/astro/blum/PhD/work/MeVDM_JUNO/test_convolution/Compare_Amplitude_in_Stheo/difference.txt",
           S_difference, fmt='%1.5e',
           header="Difference between the normalized spectra of 'signal_DMmass30_100keV_140events.txt' and \n"
                  "'signal_DMmass30_100keV.txt' in 1/MeV (difference = S_140event_norm - S_1event_norm).\n"
                  "(calculated with script compare_amplitude_in_Stheo.py)")

h1 = plt.figure(1)
plt.step(E_vis, S_vis_140events, where='mid', label='visible spectrum with {0:.2f} events'.format(N_140events))
plt.step(E_vis, S_vis_1event, where='mid', label='visible spectrum with {0:.2f} events'.format(N_1event))
plt.ylim(ymin=0)
plt.xlabel("visible energy in MeV")
plt.ylabel("visible spectrum in 1/MeV")
plt.title("Comparison of spectra for DM mass = 30 MeV (compare_amplitude_in_Stheo.py)")
plt.grid()
plt.legend()

h2 = plt.figure(2)
plt.step(E_vis, S_vis_140events_norm, where='mid',
         label='normalized visible spectrum (before {0:.2f} events)'.format(N_140events))
plt.step(E_vis, S_vis_1event_norm, where='mid', linestyle='--',
         label='normalized visible spectrum (before {0:.2f} events)'.format(N_1event))
plt.ylim(ymin=0)
plt.xlabel("visible energy in MeV")
plt.ylabel("visible spectrum in 1/MeV")
plt.title("Comparison of normalized spectra for DM mass = 30 MeV (compare_amplitude_in_Stheo.py)")
plt.grid()
plt.legend()

h3 = plt.figure(3)
plt.step(E_vis, S_difference, where='mid',
         label='difference of normalized spectra (S_140events_norm - S_1event_norm)')
plt.xlabel("visible energy in MeV")
plt.ylabel("difference of visible spectrum in 1/MeV")
plt.title("Difference of normalized spectra for DM mass = 30 MeV (compare_amplitude_in_Stheo.py)")
plt.grid()
plt.legend()

plt.show()
