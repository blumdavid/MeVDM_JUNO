""" script to display the signal spectrum of different DM masses without PSD and with PSD.

"""
import numpy as np
from matplotlib import pyplot as plt

# flag if spectra without or with PSD should be displayed:
FLAG_WITHOUT_PSD = True

# path, where signal spectra are saved:
path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
               "PSD_350ns_600ns_400000atmoNC_1MeV/"

if FLAG_WITHOUT_PSD:
    file_signal_20 = path_spectra + "signal_DMmass20_bin1000keV.txt"
    file_signal_info_20 = path_spectra + "signal_info_DMmass20_bin1000keV.txt"

    file_signal_30 = path_spectra + "signal_DMmass30_bin1000keV.txt"
    file_signal_40 = path_spectra + "signal_DMmass40_bin1000keV.txt"
    file_signal_50 = path_spectra + "signal_DMmass50_bin1000keV.txt"
    file_signal_60 = path_spectra + "signal_DMmass60_bin1000keV.txt"
    file_signal_70 = path_spectra + "signal_DMmass70_bin1000keV.txt"
    file_signal_80 = path_spectra + "signal_DMmass80_bin1000keV.txt"
    file_signal_90 = path_spectra + "signal_DMmass90_bin1000keV.txt"
    file_signal_100 = path_spectra + "signal_DMmass100_bin1000keV.txt"

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
    signal_30 = np.loadtxt(file_signal_30)
    signal_40 = np.loadtxt(file_signal_40)
    signal_50 = np.loadtxt(file_signal_50)
    signal_60 = np.loadtxt(file_signal_60)
    signal_70 = np.loadtxt(file_signal_70)
    signal_80 = np.loadtxt(file_signal_80)
    signal_90 = np.loadtxt(file_signal_90)
    signal_100 = np.loadtxt(file_signal_100)

    N_20 = np.sum(signal_20)
    N_30 = np.sum(signal_30)
    N_40 = np.sum(signal_40)
    N_50 = np.sum(signal_50)
    N_60 = np.sum(signal_60)
    N_70 = np.sum(signal_70)
    N_80 = np.sum(signal_80)
    N_90 = np.sum(signal_90)
    N_100 = np.sum(signal_100)

    h1 = plt.figure(1, figsize=(10, 5))
    plt.semilogy(E_visible, signal_20, color='grey', drawstyle="steps", label='$m_{DM}$ = 20 MeV ($N_S$'+' = {0:.2f})'
                 .format(N_20))
    plt.semilogy(E_visible, signal_30, color='black', drawstyle="steps", label='$m_{DM}$ = 30 MeV ($N_S$'+' = {0:.2f})'
                 .format(N_30))
    plt.semilogy(E_visible, signal_40, color='orange', drawstyle="steps", label='$m_{DM}$ = 40 MeV ($N_S$'+' = {0:.2f})'
                 .format(N_40))
    plt.semilogy(E_visible, signal_50, color='red', drawstyle="steps", label='$m_{DM}$ = 50 MeV ($N_S$'+' = {0:.2f})'
                 .format(N_50))
    plt.semilogy(E_visible, signal_60, color='limegreen', drawstyle="steps",
                 label='$m_{DM}$ = 60 MeV ($N_S$'+' = {0:.2f})'.format(N_60))
    plt.semilogy(E_visible, signal_70, color='darkgreen', drawstyle="steps",
                 label='$m_{DM}$ = 70 MeV ($N_S$'+' = {0:.2f})'.format(N_70))
    plt.semilogy(E_visible, signal_80, color='blue', drawstyle="steps", label='$m_{DM}$ = 80 MeV ($N_S$'+' = {0:.2f})'
                 .format(N_80))
    plt.semilogy(E_visible, signal_90, color='navy', drawstyle="steps", label='$m_{DM}$ = 90 MeV ($N_S$'+' = {0:.2f})'
                 .format(N_90))
    plt.semilogy(E_visible, signal_100, color='magenta', drawstyle="steps",
                 label='$m_{DM}$ = 100 MeV ($N_S$'+' = {0:.2f})'.format(N_100))
    plt.xlim(E_vis_min, E_vis_max)
    plt.ylim(10**(-2), 500)
    plt.xticks(np.arange(E_vis_min, E_vis_max+5, 5))
    plt.xlabel("Visible energy $E_{vis}$ in MeV", fontsize=12)
    plt.ylabel("Expected spectrum $\\frac{dN_S}{dE_{vis}}$ in events/bin" +
               "\n(bin-width = {0:.2f} MeV)".format(E_vis_bin), fontsize=12)
    plt.title("Expected energy spectrum of electron anti-neutrinos from DM self-annihilation in JUNO "
              "\n(for lifetime of {0:.0f} years,".format(t_years) +
              " $\\langle \\sigma_Av \\rangle_{natural} = 3.0 \\cdot 10^{-26}$ cm$^3$/s and $J_{avg}=5.0$)",
              fontsize=13)
    plt.legend()
    plt.grid()
    plt.show()

else:
    file_signal_20 = path_spectra + "signal_DMmass20_bin500keV_PSD.txt"
    file_signal_info_20 = path_spectra + "signal_info_DMmass20_bin500keV_PSD.txt"

    file_signal_30 = path_spectra + "signal_DMmass30_bin500keV_PSD.txt"
    file_signal_40 = path_spectra + "signal_DMmass40_bin500keV_PSD.txt"
    file_signal_50 = path_spectra + "signal_DMmass50_bin500keV_PSD.txt"
    file_signal_60 = path_spectra + "signal_DMmass60_bin500keV_PSD.txt"
    file_signal_70 = path_spectra + "signal_DMmass70_bin500keV_PSD.txt"
    file_signal_80 = path_spectra + "signal_DMmass80_bin500keV_PSD.txt"
    file_signal_90 = path_spectra + "signal_DMmass90_bin500keV_PSD.txt"
    file_signal_100 = path_spectra + "signal_DMmass100_bin500keV_PSD.txt"

    # load the information file:
    signal_info = np.loadtxt(file_signal_info_20)
    # bin width in MeV:
    E_vis_bin = signal_info[5]
    # minimal E_vis in MeV:
    E_vis_min = signal_info[3]
    # maximal E_vis in MeV:
    E_vis_max = signal_info[4]
    # visible energy in MeV (array of float):
    E_visible = np.arange(E_vis_min, E_vis_max + E_vis_bin, E_vis_bin)
    # exposure time in years:
    t_years = signal_info[6]
    # DM annihilation cross-section in cm**3/s:
    sigma_Anni = signal_info[12]

    signal_20 = np.loadtxt(file_signal_20)
    signal_30 = np.loadtxt(file_signal_30)
    signal_40 = np.loadtxt(file_signal_40)
    signal_50 = np.loadtxt(file_signal_50)
    signal_60 = np.loadtxt(file_signal_60)
    signal_70 = np.loadtxt(file_signal_70)
    signal_80 = np.loadtxt(file_signal_80)
    signal_90 = np.loadtxt(file_signal_90)
    signal_100 = np.loadtxt(file_signal_100)

    N_20 = np.sum(signal_20)
    N_30 = np.sum(signal_30)
    N_40 = np.sum(signal_40)
    N_50 = np.sum(signal_50)
    N_60 = np.sum(signal_60)
    N_70 = np.sum(signal_70)
    N_80 = np.sum(signal_80)
    N_90 = np.sum(signal_90)
    N_100 = np.sum(signal_100)

    h1 = plt.figure(1, figsize=(10, 5))
    plt.semilogy(E_visible, signal_20, color='grey', drawstyle="steps", label='$m_{DM}$ = 20 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_20))
    plt.semilogy(E_visible, signal_30, color='black', drawstyle="steps", label='$m_{DM}$ = 30 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_30))
    plt.semilogy(E_visible, signal_40, color='orange', drawstyle="steps",
                 label='$m_{DM}$ = 40 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_40))
    plt.semilogy(E_visible, signal_50, color='red', drawstyle="steps", label='$m_{DM}$ = 50 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_50))
    plt.semilogy(E_visible, signal_60, color='limegreen', drawstyle="steps",
                 label='$m_{DM}$ = 60 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_60))
    plt.semilogy(E_visible, signal_70, color='darkgreen', drawstyle="steps",
                 label='$m_{DM}$ = 70 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_70))
    plt.semilogy(E_visible, signal_80, color='blue', drawstyle="steps", label='$m_{DM}$ = 80 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_80))
    plt.semilogy(E_visible, signal_90, color='navy', drawstyle="steps", label='$m_{DM}$ = 90 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_90))
    plt.semilogy(E_visible, signal_100, color='magenta', drawstyle="steps",
                 label='$m_{DM}$ = 100 MeV ($N_S$' + ' = {0:.2f})'
                 .format(N_100))
    plt.xlim(E_vis_min, E_vis_max)
    plt.ylim(10 ** (-2), 500)
    plt.xticks(np.arange(E_vis_min, E_vis_max + 5, 5))
    plt.xlabel("Visible energy $E_{vis}$ in MeV", fontsize=12)
    plt.ylabel("Expected spectrum $\\frac{dN_S}{dE_{vis}}$ in events/bin" +
               "\n(bin-width = {0:.2f} MeV)".format(E_vis_bin), fontsize=12)
    plt.title("Expected energy spectrum of electron anti-neutrinos from DM self-annihilation in JUNO after PSD cut"
              "\n(for lifetime of {0:.0f} years,".format(t_years) +
              " $\\langle \\sigma_Av \\rangle_{natural} = 3.0 \\cdot 10^{-26}$ cm$^3$/s and $J_{avg}=5.0$)",
              fontsize=13)
    plt.legend()
    plt.grid()
    plt.show()



