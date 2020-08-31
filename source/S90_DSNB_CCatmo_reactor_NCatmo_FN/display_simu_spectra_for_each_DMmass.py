""" Script to display the simulated spectrum (signal spectrum, DSNB background, CCatmo background, Reactor background,
    NCatmo background and fast neutron) for each DM mass.

    DM masses from 20 MeV to 60 MeV are displayed in 1 plot with 8 subplots (50 MeV is skipped,
    because it is displayed in the total spectrum already)

    DM masses from 65 MeV to 100 MeV are displayed in 1 plot with 8 subplots
"""
import numpy as np
from matplotlib import pyplot as plt

path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
               "PSD_350ns_600ns_400000atmoNC_1MeV/"

path_spectra_NC = ("/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/"
                   "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/"
                   "test_350ns_600ns_400000atmoNC_1MeV/")

""" set the file names, where the simulated spectra after PSD are saved: """
signal_20_PSD = np.loadtxt(path_spectra + "signal_DMmass20_bin1000keV_PSD.txt")
signal_info = np.loadtxt(path_spectra + "signal_info_DMmass20_bin1000keV_PSD.txt")
signal_25_PSD = np.loadtxt(path_spectra + "signal_DMmass25_bin1000keV_PSD.txt")
signal_30_PSD = np.loadtxt(path_spectra + "signal_DMmass30_bin1000keV_PSD.txt")
signal_35_PSD = np.loadtxt(path_spectra + "signal_DMmass35_bin1000keV_PSD.txt")
signal_40_PSD = np.loadtxt(path_spectra + "signal_DMmass40_bin1000keV_PSD.txt")
signal_45_PSD = np.loadtxt(path_spectra + "signal_DMmass45_bin1000keV_PSD.txt")
signal_50_PSD = np.loadtxt(path_spectra + "signal_DMmass50_bin1000keV_PSD.txt")
signal_55_PSD = np.loadtxt(path_spectra + "signal_DMmass55_bin1000keV_PSD.txt")
signal_60_PSD = np.loadtxt(path_spectra + "signal_DMmass60_bin1000keV_PSD.txt")
signal_65_PSD = np.loadtxt(path_spectra + "signal_DMmass65_bin1000keV_PSD.txt")
signal_70_PSD = np.loadtxt(path_spectra + "signal_DMmass70_bin1000keV_PSD.txt")
signal_75_PSD = np.loadtxt(path_spectra + "signal_DMmass75_bin1000keV_PSD.txt")
signal_80_PSD = np.loadtxt(path_spectra + "signal_DMmass80_bin1000keV_PSD.txt")
signal_85_PSD = np.loadtxt(path_spectra + "signal_DMmass85_bin1000keV_PSD.txt")
signal_90_PSD = np.loadtxt(path_spectra + "signal_DMmass90_bin1000keV_PSD.txt")
signal_95_PSD = np.loadtxt(path_spectra + "signal_DMmass95_bin1000keV_PSD.txt")
signal_100_PSD = np.loadtxt(path_spectra + "signal_DMmass100_bin1000keV_PSD.txt")

DSNB_PSD = np.loadtxt(path_spectra + "DSNB_bin1000keV_f027_PSD.txt")
CCatmo_p_PSD = np.loadtxt(path_spectra + "CCatmo_onlyP_Osc1_bin1000keV_PSD.txt")
CCatmo_C12_PSD = np.loadtxt(path_spectra + "CCatmo_onlyC12_Osc1_bin1000keV_PSD.txt")
NCatmo_PSD = np.loadtxt(path_spectra_NC + "NCatmo_onlyC12_wPSD99_bin1000keV_fit.txt")

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

""" calculate number of events from spectrum after PSD: """
N_signal_20_PSD = np.sum(signal_20_PSD)
N_signal_25_PSD = np.sum(signal_25_PSD)
N_signal_30_PSD = np.sum(signal_30_PSD)
N_signal_35_PSD = np.sum(signal_35_PSD)
N_signal_40_PSD = np.sum(signal_40_PSD)
N_signal_45_PSD = np.sum(signal_45_PSD)
N_signal_50_PSD = np.sum(signal_50_PSD)
N_signal_55_PSD = np.sum(signal_55_PSD)
N_signal_60_PSD = np.sum(signal_60_PSD)
N_signal_65_PSD = np.sum(signal_65_PSD)
N_signal_70_PSD = np.sum(signal_70_PSD)
N_signal_75_PSD = np.sum(signal_75_PSD)
N_signal_80_PSD = np.sum(signal_80_PSD)
N_signal_85_PSD = np.sum(signal_85_PSD)
N_signal_90_PSD = np.sum(signal_90_PSD)
N_signal_95_PSD = np.sum(signal_95_PSD)
N_signal_100_PSD = np.sum(signal_100_PSD)

N_DSNB_PSD = np.sum(DSNB_PSD)
N_CCatmo_p_PSD = np.sum(CCatmo_p_PSD)
N_CCatmo_C12_PSD = np.sum(CCatmo_C12_PSD)
N_NCatmo_PSD = np.sum(NCatmo_PSD)

""" total spectrum per bin: """
spectrum_20_PSD = signal_20_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_25_PSD = signal_25_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_30_PSD = signal_30_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_35_PSD = signal_35_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_40_PSD = signal_40_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_45_PSD = signal_45_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_50_PSD = signal_50_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_55_PSD = signal_55_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_60_PSD = signal_60_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_65_PSD = signal_65_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_70_PSD = signal_70_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_75_PSD = signal_75_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_80_PSD = signal_80_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_85_PSD = signal_85_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_90_PSD = signal_90_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_95_PSD = signal_95_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD
spectrum_100_PSD = signal_100_PSD + DSNB_PSD + CCatmo_p_PSD + CCatmo_C12_PSD + NCatmo_PSD

""" create first plot: """
fig, axs = plt.subplots(4, 2, num=1, sharex='all', sharey='all', figsize=(11, 12))
# 20 MeV:
axs[0, 0].semilogy(E_visible, signal_20_PSD, 'r-', label='20 MeV (N = {0:.1f})'.format(N_signal_20_PSD),
                   drawstyle="steps")
axs[0, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[0, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[0, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[0, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[0, 0].semilogy(E_visible, spectrum_20_PSD, 'k-', drawstyle="steps")
axs[0, 0].legend()
axs[0, 0].grid()
# 25 MeV:
axs[0, 1].semilogy(E_visible, signal_25_PSD, 'r-', label='25 MeV (N = {0:.1f})'.format(N_signal_25_PSD),
                   drawstyle="steps")
axs[0, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[0, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[0, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[0, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[0, 1].semilogy(E_visible, spectrum_25_PSD, 'k-', drawstyle="steps")
axs[0, 1].legend()
axs[0, 1].grid()
# 30 MeV:
axs[1, 0].semilogy(E_visible, signal_30_PSD, 'r-', label='30 MeV (N = {0:.1f})'.format(N_signal_30_PSD),
                   drawstyle="steps")
axs[1, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[1, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[1, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[1, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[1, 0].semilogy(E_visible, spectrum_30_PSD, 'k-', drawstyle="steps")
axs[1, 0].legend()
axs[1, 0].grid()
# 35 MeV:
axs[1, 1].semilogy(E_visible, signal_35_PSD, 'r-', label='35 MeV (N = {0:.1f})'.format(N_signal_35_PSD),
                   drawstyle="steps")
axs[1, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[1, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[1, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[1, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[1, 1].semilogy(E_visible, spectrum_35_PSD, 'k-', drawstyle="steps")
axs[1, 1].legend()
axs[1, 1].grid()
# 40 MeV:
axs[2, 0].semilogy(E_visible, signal_40_PSD, 'r-', label='40 MeV (N = {0:.1f})'.format(N_signal_40_PSD),
                   drawstyle="steps")
axs[2, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[2, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[2, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[2, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[2, 0].semilogy(E_visible, spectrum_40_PSD, 'k-', drawstyle="steps")
axs[2, 0].legend()
axs[2, 0].grid()
# 45 MeV:
axs[2, 1].semilogy(E_visible, signal_45_PSD, 'r-', label='45 MeV (N = {0:.1f})'.format(N_signal_45_PSD),
                   drawstyle="steps")
axs[2, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[2, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[2, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[2, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[2, 1].semilogy(E_visible, spectrum_45_PSD, 'k-', drawstyle="steps")
axs[2, 1].legend()
axs[2, 1].grid()
# 55 MeV:
axs[3, 0].semilogy(E_visible, signal_55_PSD, 'r-', label='55 MeV (N = {0:.1f})'.format(N_signal_55_PSD),
                   drawstyle="steps")
axs[3, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[3, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[3, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[3, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[3, 0].semilogy(E_visible, spectrum_55_PSD, 'k-', drawstyle="steps")
axs[3, 0].legend()
axs[3, 0].grid()
# 60 MeV:
axs[3, 1].semilogy(E_visible, signal_60_PSD, 'r-', label='60 MeV (N = {0:.1f})'.format(N_signal_60_PSD),
                   drawstyle="steps")
axs[3, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[3, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[3, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[3, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[3, 1].semilogy(E_visible, spectrum_60_PSD, 'k-', drawstyle="steps")
axs[3, 1].legend()
axs[3, 1].grid()


# set x and y limits:
for ax in axs.flat:
    ax.set_xlim(E_vis_min, E_vis_max)
    ax.set_ylim(10**(-2), 10)

# set x- and y-label:
for ax in axs.flat:
    ax.set(xlabel="Visible energy in MeV",
           ylabel="dN/dE in events/bin\n(bin-width = {0:.1f} MeV)".format(E_vis_bin))

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.suptitle("Expected spectrum in JUNO after {0:.0f} years of lifetime for several DM masses after PSD"
             .format(t_years), fontsize=13, y=0.95)

""" create second plot: """
fig, axs = plt.subplots(4, 2, num=2, sharex='all', sharey='all', figsize=(11, 12))
# 65 MeV:
axs[0, 0].semilogy(E_visible, signal_65_PSD, 'r-', label='65 MeV (N = {0:.1f})'.format(N_signal_65_PSD),
                   drawstyle="steps")
axs[0, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[0, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[0, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[0, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[0, 0].semilogy(E_visible, spectrum_65_PSD, 'k-', drawstyle="steps")
axs[0, 0].legend()
axs[0, 0].grid()
# 70 MeV:
axs[0, 1].semilogy(E_visible, signal_70_PSD, 'r-', label='70 MeV (N = {0:.1f})'.format(N_signal_70_PSD),
                   drawstyle="steps")
axs[0, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[0, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[0, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[0, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[0, 1].semilogy(E_visible, spectrum_70_PSD, 'k-', drawstyle="steps")
axs[0, 1].legend()
axs[0, 1].grid()
# 75 MeV:
axs[1, 0].semilogy(E_visible, signal_75_PSD, 'r-', label='75 MeV (N = {0:.1f})'.format(N_signal_75_PSD),
                   drawstyle="steps")
axs[1, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[1, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[1, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[1, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[1, 0].semilogy(E_visible, spectrum_75_PSD, 'k-', drawstyle="steps")
axs[1, 0].legend()
axs[1, 0].grid()
# 80 MeV:
axs[1, 1].semilogy(E_visible, signal_80_PSD, 'r-', label='80 MeV (N = {0:.1f})'.format(N_signal_80_PSD),
                   drawstyle="steps")
axs[1, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[1, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[1, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[1, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[1, 1].semilogy(E_visible, spectrum_80_PSD, 'k-', drawstyle="steps")
axs[1, 1].legend()
axs[1, 1].grid()
# 85 MeV:
axs[2, 0].semilogy(E_visible, signal_85_PSD, 'r-', label='85 MeV (N = {0:.1f})'.format(N_signal_85_PSD),
                   drawstyle="steps")
axs[2, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[2, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[2, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[2, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[2, 0].semilogy(E_visible, spectrum_85_PSD, 'k-', drawstyle="steps")
axs[2, 0].legend()
axs[2, 0].grid()
# 90 MeV:
axs[2, 1].semilogy(E_visible, signal_90_PSD, 'r-', label='90 MeV (N = {0:.1f})'.format(N_signal_90_PSD),
                   drawstyle="steps")
axs[2, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[2, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[2, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[2, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[2, 1].semilogy(E_visible, spectrum_90_PSD, 'k-', drawstyle="steps")
axs[2, 1].legend()
axs[2, 1].grid()
# 95 MeV:
axs[3, 0].semilogy(E_visible, signal_95_PSD, 'r-', label='95 MeV (N = {0:.1f})'.format(N_signal_95_PSD),
                   drawstyle="steps")
axs[3, 0].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[3, 0].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[3, 0].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[3, 0].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[3, 0].semilogy(E_visible, spectrum_95_PSD, 'k-', drawstyle="steps")
axs[3, 0].legend()
axs[3, 0].grid()
# 100 MeV:
axs[3, 1].semilogy(E_visible, signal_100_PSD, 'r-', label='100 MeV (N = {0:.1f})'.format(N_signal_100_PSD),
                   drawstyle="steps")
axs[3, 1].semilogy(E_visible, DSNB_PSD, 'b-', drawstyle="steps")
axs[3, 1].semilogy(E_visible, CCatmo_p_PSD, 'g-', drawstyle="steps")
axs[3, 1].semilogy(E_visible, CCatmo_C12_PSD, 'g--', drawstyle="steps")
axs[3, 1].semilogy(E_visible, NCatmo_PSD, color='orange', drawstyle="steps")
axs[3, 1].semilogy(E_visible, spectrum_100_PSD, 'k-', drawstyle="steps")
axs[3, 1].legend()
axs[3, 1].grid()


# set x and y limits:
for ax in axs.flat:
    ax.set_xlim(E_vis_min, E_vis_max)
    ax.set_ylim(10**(-2), 10)

# set x- and y-label:
for ax in axs.flat:
    ax.set(xlabel="Visible energy in MeV",
           ylabel="dN/dE in events/bin\n(bin-width = {0:.1f} MeV)".format(E_vis_bin))

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.suptitle("Expected spectrum in JUNO after {0:.0f} years of lifetime for several DM masses after PSD"
             .format(t_years), fontsize=13, y=0.95)

plt.show()









