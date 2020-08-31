""" Script to display the simulated spectrum (signal spectrum, DSNB background, CCatmo background, Reactor background,
    NCatmo background and fast neutron)
"""
import numpy as np
from matplotlib import pyplot as plt

# PSD parameter:
# test_10to20_20to30_30to40_40to100_200000atmoNC_1MeV:
# IBD_supp_total = 11.60
# NC_supp_total = 99.03
# FN_supp_total = 99.94

# test_350ns_600ns_400000atmoNC_1MeV:
IBD_eff_total = 88.21
NC_eff_total = 0.92

# set DM mass in MeV:
# masses = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
masses = np.array([50])

# path to the directory, where the files of the simulated spectra are saved:
# path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"

# path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
#                "PSD_350ns_600ns_400000atmoNC_1MeV/"

path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
               "PSD_350ns_600ns_400000atmoNC_1MeV_54_4years//"

# path_spectra_NC = ("/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/"
#                    "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/"
#                    "test_10to20_20to30_30to40_40to100_200000atmoNC_1MeV/")

path_spectra_NC = ("/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/"
                   "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/"
                   "test_350ns_600ns_400000atmoNC_1MeV/")

path_spectra_NC = path_spectra

for dm_mass in masses:

    # set the file names, where the simulated spectra are saved:

    # bin width 500 keV:
    """   
    # file_signal = path_spectra + "0signal_bin500keV.txt"
    # file_signal_info = path_spectra + "0signal_info_bin500keV.txt"
    file_signal = path_spectra + "signal_DMmass{0:d}_bin500keV.txt".format(dm_mass)
    file_signal_info = path_spectra + "signal_info_DMmass{0:d}_bin500keV.txt".format(dm_mass)
    file_DSNB = path_spectra + "DSNB_bin500keV_f027.txt"
    file_DSNB_info = path_spectra + "DSNB_info_bin500keV_f027.txt"
    file_CCatmo_p = path_spectra + "CCatmo_onlyP_Osc1_bin500keV.txt"
    file_CCatmo_info = path_spectra + "CCatmo_onlyP_info_Osc1_bin500keV.txt"
    file_CCatmo_C12 = path_spectra + "CCatmo_onlyC12_Osc1_bin500keV.txt"
    file_reactor = path_spectra + "Reactor_NH_power36_bin500keV.txt"
    file_reactor_info = path_spectra + "Reactor_info_NH_power36_bin500keV.txt"
    file_NCatmo = path_spectra_NC + "NCatmo_onlyC12_woPSD_bin500keV.txt"
    file_NCatmo_info = path_spectra_NC + "NCatmo_info_onlyC12_woPSD_bin500keV.txt"
    file_fastneutron = path_spectra + "fast_neutron_33events_bin500keV.txt"
    file_fastneutron_info = path_spectra + "fast_neutron_info_33events_bin500keV.txt"

    # set the file names, where the simulated spectra after PSD are saved:
    # file_signal_PSD = path_spectra + "0signal_bin500keV.txt"
    # file_signal_info_PSD = path_spectra + "0signal_info_bin500keV.txt"
    file_signal_PSD = path_spectra + "signal_DMmass{0:d}_bin500keV_PSD.txt".format(dm_mass)
    file_signal_info_PSD = path_spectra + "signal_info_DMmass{0:d}_bin500keV_PSD.txt".format(dm_mass)
    file_DSNB_PSD = path_spectra + "DSNB_bin500keV_f027_PSD.txt"
    file_DSNB_info_PSD = path_spectra + "DSNB_info_bin500keV_f027_PSD.txt"
    file_CCatmo_p_PSD = path_spectra + "CCatmo_onlyP_Osc1_bin500keV_PSD.txt"
    file_CCatmo_p_info_PSD = path_spectra + "CCatmo_onlyP_info_Osc1_bin500keV_PSD.txt"
    file_CCatmo_C12_PSD = path_spectra + "CCatmo_onlyC12_Osc1_bin500keV_PSD.txt"
    file_reactor_PSD = path_spectra + "Reactor_NH_power36_bin500keV_PSD.txt"
    file_reactor_info_PSD = path_spectra + "Reactor_info_NH_power36_bin500keV_PSD.txt"
    file_NCatmo_PSD = path_spectra_NC + "NCatmo_onlyC12_wPSD99_bin500keV.txt"
    file_NCatmo_info_PSD = path_spectra_NC + "NCatmo_info_onlyC12_wPSD99_bin500keV.txt"
    file_fastneutron_PSD = path_spectra + "fast_neutron_33events_bin500keV_PSD.txt"
    file_fastneutron_info_PSD = path_spectra + "fast_neutron_info_33events_bin500keV_PSD.txt"
    """

    # bin width 1 MeV:

    # file_signal = path_spectra + "0signal_bin1000keV.txt"
    # file_signal_info = path_spectra + "0signal_info_bin1000keV.txt"
    file_signal = path_spectra + "signal_DMmass{0:d}_bin1000keV.txt".format(dm_mass)
    file_signal_info = path_spectra + "signal_info_DMmass{0:d}_bin1000keV.txt".format(dm_mass)
    file_DSNB = path_spectra + "DSNB_bin1000keV_f027.txt"
    file_DSNB_info = path_spectra + "DSNB_info_bin1000keV_f027.txt"
    file_CCatmo_p = path_spectra + "CCatmo_onlyP_Osc1_bin1000keV.txt"
    file_CCatmo_info = path_spectra + "CCatmo_onlyP_info_Osc1_bin1000keV.txt"
    file_CCatmo_C12 = path_spectra + "CCatmo_onlyC12_Osc1_bin1000keV.txt"
    file_reactor = path_spectra + "Reactor_NH_power36_bin1000keV.txt"
    file_reactor_info = path_spectra + "Reactor_info_NH_power36_bin1000keV.txt"
    file_NCatmo = path_spectra_NC + "NCatmo_onlyC12_woPSD_bin1000keV.txt"
    file_NCatmo_info = path_spectra_NC + "NCatmo_info_onlyC12_woPSD_bin1000keV.txt"
    file_fastneutron = path_spectra + "fast_neutron_33events_bin1000keV.txt"
    file_fastneutron_info = path_spectra + "fast_neutron_info_33events_bin1000keV.txt"

    # set the file names, where the simulated spectra after PSD are saved:
    # file_signal_PSD = path_spectra + "0signal_bin1000keV.txt"
    # file_signal_info_PSD = path_spectra + "0signal_info_bin1000keV.txt"
    file_signal_PSD = path_spectra + "signal_DMmass{0:d}_bin1000keV_PSD.txt".format(dm_mass)
    file_signal_info_PSD = path_spectra + "signal_info_DMmass{0:d}_bin1000keV_PSD.txt".format(dm_mass)
    file_DSNB_PSD = path_spectra + "DSNB_bin1000keV_f027_PSD.txt"
    file_DSNB_info_PSD = path_spectra + "DSNB_info_bin1000keV_f027_PSD.txt"
    file_CCatmo_p_PSD = path_spectra + "CCatmo_onlyP_Osc1_bin1000keV_PSD.txt"
    file_CCatmo_p_info_PSD = path_spectra + "CCatmo_onlyP_info_Osc1_bin1000keV_PSD.txt"
    file_CCatmo_C12_PSD = path_spectra + "CCatmo_onlyC12_Osc1_bin1000keV_PSD.txt"
    file_reactor_PSD = path_spectra + "Reactor_NH_power36_bin1000keV_PSD.txt"
    file_reactor_info_PSD = path_spectra + "Reactor_info_NH_power36_bin1000keV_PSD.txt"
    file_NCatmo_PSD = path_spectra_NC + "NCatmo_onlyC12_wPSD99_bin1000keV_fit.txt"
    file_NCatmo_info_PSD = path_spectra_NC + "NCatmo_info_onlyC12_wPSD99_bin1000keV.txt"
    file_fastneutron_PSD = path_spectra + "fast_neutron_33events_bin1000keV_PSD.txt"
    file_fastneutron_info_PSD = path_spectra + "fast_neutron_info_33events_bin1000keV_PSD.txt"

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
    # DM annihilation cross-section in cm**3/s:
    sigma_Anni = signal_info[12]

    # spectra in electron-neutrinos/bin) (np.array of float):
    signal_per_bin = np.loadtxt(file_signal)
    DSNB_per_bin = np.loadtxt(file_DSNB)
    CCatmo_p_per_bin = np.loadtxt(file_CCatmo_p)
    CCatmo_C12_per_bin = np.loadtxt(file_CCatmo_C12)
    reactor_per_bin = np.loadtxt(file_reactor)
    NCatmo_per_bin = np.loadtxt(file_NCatmo)
    fast_neutron_per_bin = np.loadtxt(file_fastneutron)

    # spectra in events/bin after PSD:
    signal_per_bin_PSD = np.loadtxt(file_signal_PSD)
    DSNB_per_bin_PSD = np.loadtxt(file_DSNB_PSD)
    CCatmo_p_per_bin_PSD = np.loadtxt(file_CCatmo_p_PSD)
    CCatmo_C12_per_bin_PSD = np.loadtxt(file_CCatmo_C12_PSD)
    reactor_per_bin_PSD = np.loadtxt(file_reactor_PSD)
    NCatmo_per_bin_PSD = np.loadtxt(file_NCatmo_PSD)
    fast_neutron_per_bin_PSD = np.loadtxt(file_fastneutron_PSD)

    # calculate number of events from spectrum:
    N_signal = np.sum(signal_per_bin)
    N_DSNB = np.sum(DSNB_per_bin)
    N_CCatmo_p = np.sum(CCatmo_p_per_bin)
    N_CCatmo_C12 = np.sum(CCatmo_C12_per_bin)
    N_reactor = np.sum(reactor_per_bin)
    N_NCatmo = np.sum(NCatmo_per_bin)
    N_fastneutron = np.sum(fast_neutron_per_bin)

    # calculate number of events from spectrum after PSD:
    N_signal_PSD = np.sum(signal_per_bin_PSD)
    N_DSNB_PSD = np.sum(DSNB_per_bin_PSD)
    N_CCatmo_p_PSD = np.sum(CCatmo_p_per_bin_PSD)
    N_CCatmo_C12_PSD = np.sum(CCatmo_C12_per_bin_PSD)
    N_reactor_PSD = np.sum(reactor_per_bin_PSD)
    N_NCatmo_PSD = np.sum(NCatmo_per_bin_PSD)
    N_fastneutron_PSD = np.sum(fast_neutron_per_bin_PSD)

    # total spectrum in electron-antineutrinos/bin:
    # spectrum_per_bin = (signal_per_bin + DSNB_per_bin + CCatmo_p_per_bin + CCatmo_C12_per_bin + reactor_per_bin +
    #                     NCatmo_per_bin + fast_neutron_per_bin)
    spectrum_per_bin_PSD = (signal_per_bin_PSD + DSNB_per_bin_PSD + CCatmo_p_per_bin_PSD + CCatmo_C12_per_bin_PSD +
                            NCatmo_per_bin_PSD)
    # spectrum_per_bin = (DSNB_per_bin + CCatmo_p_per_bin + CCatmo_C12_per_bin + reactor_per_bin + NCatmo_per_bin +
    #                     fast_neutron_per_bin)
    # spectrum_per_bin_PSD = DSNB_per_bin_PSD + CCatmo_p_per_bin_PSD + CCatmo_C12_per_bin_PSD + NCatmo_per_bin_PSD

    h1 = plt.figure(1, figsize=(11, 6))
    plt.semilogy(E_visible, signal_per_bin, 'r-', label='signal from DM annihilation for '
                 '$<\\sigma_Av>=${0:.1e}$cm^3/s$\nexpected events = {1:.1f}'.format(sigma_Anni, N_signal),
                 drawstyle="steps")
    plt.semilogy(E_visible, DSNB_per_bin, 'b-', label='DSNB (expected events = {0:.1f})'
                 .format(N_DSNB), drawstyle="steps")
    plt.semilogy(E_visible, reactor_per_bin, 'c-', label='reactor bkg (expected events = {0:.1f})'
                 .format(N_reactor), drawstyle="steps")
    plt.semilogy(E_visible, CCatmo_p_per_bin, 'g-',
                 label='atmo. CC bkg on p (expected events = {0:.1f})'
                 .format(N_CCatmo_p), drawstyle="steps")
    plt.semilogy(E_visible, CCatmo_C12_per_bin, 'g--',
                 label='atmo. CC bkg on C12 (expected events = {0:.1f})'
                 .format(N_CCatmo_C12), drawstyle="steps")
    plt.semilogy(E_visible, NCatmo_per_bin, color='orange', drawstyle="steps",
                 label='atmo. NC bkg (expected events = {0:.1f})'.format(N_NCatmo))
    plt.semilogy(E_visible, fast_neutron_per_bin, 'm-', label='fast neutron bkg (expected events = {0:.1f})'
                 .format(N_fastneutron), drawstyle="steps")
    plt.semilogy(E_visible, spectrum_per_bin, 'k-', label='total spectrum', drawstyle="steps")
    plt.xlim(E_vis_min, E_vis_max)
    plt.ylim(10**(-2), 1000)
    plt.xlabel("Visible energy in MeV", fontsize=12)
    plt.ylabel("Expected spectrum dN/dE in events/bin\n(bin-width = {0:.1f} MeV)".format(E_vis_bin), fontsize=12)
    plt.title("Expected spectrum in JUNO after {0:.0f} years of lifetime for {1:d} MeV Dark Matter signal"
              .format(t_years, dm_mass), fontsize=13)
    # plt.title("Expected background spectrum in JUNO after {0:.0f} years".format(t_years))
    plt.legend()
    plt.grid()
    # plt.savefig(path_spectra + "spectrum_{0:d}MeV.png".format(dm_mass))
    # plt.close()

    h2 = plt.figure(2, figsize=(11, 6))
    plt.semilogy(E_visible, signal_per_bin_PSD, 'r-', drawstyle="steps", label='signal from DM annihilation for '
                 '$<\\sigma_Av>=${0:.1e}$cm^3/s$\nexpected events = {1:.1f}'.format(sigma_Anni, N_signal_PSD))
    plt.semilogy(E_visible, DSNB_per_bin_PSD, 'b-', label='DSNB (expected events = {0:.1f})'
                 .format(N_DSNB_PSD), drawstyle="steps")
    # plt.semilogy(E_visible, reactor_per_bin_PSD, 'c-', label='reactor bkg (expected events = {0:.1f})'
    #              .format(N_reactor_PSD), drawstyle="steps")
    plt.semilogy(E_visible, CCatmo_p_per_bin_PSD, 'g-', label='atmo. CC bkg on p (expected events = {0:.1f})'
                 .format(N_CCatmo_p_PSD), drawstyle="steps")
    plt.semilogy(E_visible, CCatmo_C12_per_bin_PSD, 'g--',
                 label='atmo. CC bkg on C12 (expected events = {0:.1f})'
                 .format(N_CCatmo_C12_PSD), drawstyle="steps")
    plt.semilogy(E_visible, NCatmo_per_bin_PSD, color='orange', drawstyle="steps",
                 label='atmo. NC bkg (expected events = {0:.1f})'.format(N_NCatmo_PSD))
    # plt.semilogy(E_visible, fast_neutron_per_bin_PSD, 'm-', label='fast neutron bkg (expected events = {0:.1f})'
    #              .format(N_fastneutron_PSD), drawstyle="steps")
    plt.semilogy(E_visible, spectrum_per_bin_PSD, 'k-', label='total spectrum', drawstyle="steps")
    plt.xlim(E_vis_min, E_vis_max)
    plt.ylim(10**(-2), 1000)
    plt.xlabel("Visible energy in MeV", fontsize=12)
    plt.ylabel("Expected spectrum dN/dE in events/bin\n(bin-width = {0:.1f} MeV)".format(E_vis_bin), fontsize=12)
    # plt.title("Expected spectrum in JUNO after {0:.0f} years of lifetime for {1:d} MeV Dark Matter signal after PSD\n"
    #           "(IBD eff. = {2:.2f} %, NC eff. = {3:.2f} %)"
    #           .format(t_years, dm_mass, IBD_eff_total, NC_eff_total), fontsize=13)
    plt.title("Expected background-only spectrum in JUNO after {0:.0f} years after PSD\n"
              "(IBD eff. = {1:.2f} %, NC eff. = {2:.2f} %)"
              .format(t_years, IBD_eff_total, NC_eff_total), fontsize=13)
    plt.legend()
    plt.grid()
    # plt.savefig(path_spectra + "spectrum_{0:d}MeV_PSD.png".format(dm_mass))
    # plt.close()

    plt.show()
