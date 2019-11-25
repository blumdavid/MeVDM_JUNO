""" Script to display the simulated spectrum (signal spectrum, DSNB background, CCatmo background, Reactor background,
    NCatmo background and fast neutron)
"""
import numpy as np
from matplotlib import pyplot as plt

# PSD parameter:
IBD_supp = 11.080
NC_supp = 99.0
FN_supp = 99.921
# set DM mass in MeV:
masses = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
# masses = np.array([0])

# path to the directory, where the files of the simulated spectra are saved:
path_spectra = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN/"
path_spectra_NC = ("/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/DCR_results_16000mm_10MeVto100MeV_"
                   "500nsto1ms_mult1_2400PEto3400PE_dist500mm_R17700mm_PSD{0:.0f}/".format(NC_supp))

for dm_mass in masses:

    # set the file names, where the simulated spectra are saved:
    # file_signal = path_spectra + "0signal_bin500keV.txt"
    # file_signal_info = path_spectra + "0signal_info_bin500keV.txt"
    file_signal = path_spectra + "signal_DMmass{0:d}_bin500keV.txt".format(dm_mass)
    file_signal_info = path_spectra + "signal_info_DMmass{0:d}_bin500keV.txt".format(dm_mass)
    file_DSNB = path_spectra + "DSNB_EmeanNuXbar22_bin500keV.txt"
    file_DSNB_info = path_spectra + "DSNB_info_EmeanNuXbar22_bin500keV.txt"
    file_CCatmo = path_spectra + "CCatmo_total_Osc1_bin500keV.txt"
    file_CCatmo_info = path_spectra + "CCatmo_total_info_Osc1_bin500keV.txt"
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
    file_DSNB_PSD = path_spectra + "DSNB_EmeanNuXbar22_bin500keV_PSD.txt"
    file_DSNB_info_PSD = path_spectra + "DSNB_info_EmeanNuXbar22_bin500keV_PSD.txt"
    file_CCatmo_PSD = path_spectra + "CCatmo_total_Osc1_bin500keV_PSD.txt"
    file_CCatmo_info_PSD = path_spectra + "CCatmo_total_info_Osc1_bin500keV_PSD.txt"
    file_reactor_PSD = path_spectra + "Reactor_NH_power36_bin500keV_PSD.txt"
    file_reactor_info_PSD = path_spectra + "Reactor_info_NH_power36_bin500keV_PSD.txt"
    file_NCatmo_PSD = path_spectra_NC + "NCatmo_onlyC12_wPSD{0:.0f}_bin500keV.txt".format(NC_supp)
    file_NCatmo_info_PSD = path_spectra_NC + "NCatmo_info_onlyC12_wPSD{0:.0f}_bin500keV.txt".format(NC_supp)
    file_fastneutron_PSD = path_spectra + "fast_neutron_33events_bin500keV_PSD.txt"
    file_fastneutron_info_PSD = path_spectra + "fast_neutron_info_33events_bin500keV_PSD.txt"

    # load the spectra (in electron-antineutrinos/MeV) (np.array of float):
    DSNB_per_MeV = np.loadtxt(file_DSNB)
    CCatmo_per_MeV = np.loadtxt(file_CCatmo)
    reactor_per_MeV = np.loadtxt(file_reactor)

    # load spectra after PSD in events/MeV:
    DSNB_per_MeV_PSD = np.loadtxt(file_DSNB_PSD)
    CCatmo_per_MeV_PSD = np.loadtxt(file_CCatmo_PSD)
    reactor_per_MeV_PSD = np.loadtxt(file_reactor_PSD)

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
    DSNB_per_bin = DSNB_per_MeV * E_vis_bin
    CCatmo_per_bin = CCatmo_per_MeV * E_vis_bin
    reactor_per_bin = reactor_per_MeV * E_vis_bin
    NCatmo_per_bin = np.loadtxt(file_NCatmo)
    fast_neutron_per_bin = np.loadtxt(file_fastneutron)

    # spectra in events/bin after PSD:
    signal_per_bin_PSD = np.loadtxt(file_signal_PSD)
    DSNB_per_bin_PSD = DSNB_per_MeV_PSD * E_vis_bin
    CCatmo_per_bin_PSD = CCatmo_per_MeV_PSD * E_vis_bin
    reactor_per_bin_PSD = reactor_per_MeV_PSD * E_vis_bin
    NCatmo_per_bin_PSD = np.loadtxt(file_NCatmo_PSD)
    fast_neutron_per_bin_PSD = np.loadtxt(file_fastneutron_PSD)

    # calculate number of events from spectrum:
    N_signal = np.sum(signal_per_bin)
    N_DSNB = np.sum(DSNB_per_bin)
    N_CCatmo = np.sum(CCatmo_per_bin)
    N_reactor = np.sum(reactor_per_bin)
    N_NCatmo = np.sum(NCatmo_per_bin)
    N_fastneutron = np.sum(fast_neutron_per_bin)

    # calculate number of events from spectrum after PSD:
    N_signal_PSD = np.sum(signal_per_bin_PSD)
    N_DSNB_PSD = np.sum(DSNB_per_bin_PSD)
    N_CCatmo_PSD = np.sum(CCatmo_per_bin_PSD)
    N_reactor_PSD = np.sum(reactor_per_bin_PSD)
    N_NCatmo_PSD = np.sum(NCatmo_per_bin_PSD)
    N_fastneutron_PSD = np.sum(fast_neutron_per_bin_PSD)

    # # total spectrum in electron-antineutrinos/bin:
    spectrum_per_bin = (signal_per_bin + DSNB_per_bin + CCatmo_per_bin + reactor_per_bin + NCatmo_per_bin +
                        fast_neutron_per_bin)
    spectrum_per_bin_PSD = (signal_per_bin_PSD + DSNB_per_bin_PSD + CCatmo_per_bin_PSD + reactor_per_bin_PSD +
                            NCatmo_per_bin_PSD + fast_neutron_per_bin_PSD)
    # spectrum_per_bin = (DSNB_per_bin + CCatmo_per_bin + reactor_per_bin + NCatmo_per_bin +
    #                     fast_neutron_per_bin)
    # spectrum_per_bin_PSD = (DSNB_per_bin_PSD + CCatmo_per_bin_PSD + reactor_per_bin_PSD +
    #                         NCatmo_per_bin_PSD + fast_neutron_per_bin_PSD)

    # Display the expected spectra with the settings below:
    # h1 = plt.figure(1, figsize=(15, 8))
    # plt.step(E_visible, signal_per_bin, 'r-', label='signal from DM annihilation for '
    #          '$<\\sigma_Av>=${0:.1e}$cm^3/s$\nexpected events = {1:.2f}'.format(sigma_Anni, N_signal), where='mid')
    # plt.step(E_visible, DSNB_per_bin, 'b-', label='DSNB background (expected events = {0:.2f})'
    #          .format(N_DSNB), where='mid')
    # plt.step(E_visible, reactor_per_bin, 'c-', label='reactor background (expected events = {0:.2f})'
    #          .format(N_reactor), where='mid')
    # plt.step(E_visible, CCatmo_per_bin, 'g-', label='atmospheric CC background (expected events = {0:.2f})'
    #          .format(N_CCatmo), where='mid')
    # plt.step(E_visible, NCatmo_per_bin, color='orange', label='atmospheric NC background (expected events '
    #                                                           '= {0:.2f})'.format(N_NCatmo), where='mid')
    # plt.step(E_visible, fast_neutron_per_bin, 'm-', label='fast neutron background (expected events = {0:.2f})'
    #          .format(N_fastneutron), where='mid')
    # plt.step(E_visible, spectrum_per_bin, 'k-', label='total spectrum', where='mid')
    # plt.xlim(E_vis_min, E_vis_max)
    # plt.ylim(ymin=0)
    # plt.xlabel("Visible energy in MeV")
    # plt.ylabel("Expected spectrum dN/dE in events/bin (bin-width = {0:.2f} MeV)".format(E_vis_bin))
    # plt.title("Expected spectrum in JUNO after {0:.0f} years for {1:d} MeV Dark Matter signal".format(t_years, dm_mass))
    # plt.legend()
    # plt.grid()

    # h2 = plt.figure(2, figsize=(15, 8))
    # plt.step(E_visible, signal_per_bin_PSD, 'r-', label='signal from DM annihilation for '
    #          '$<\\sigma_Av>=${0:.1e}$cm^3/s$\nexpected events = {1:.2f}'.format(sigma_Anni, N_signal_PSD), where='mid')
    # plt.step(E_visible, DSNB_per_bin_PSD, 'b-', label='DSNB background (expected events = {0:.2f})'
    #          .format(N_DSNB_PSD), where='mid')
    # plt.step(E_visible, reactor_per_bin_PSD, 'c-', label='reactor background (expected events = {0:.2f})'
    #          .format(N_reactor_PSD), where='mid')
    # plt.step(E_visible, CCatmo_per_bin_PSD, 'g-', label='atmospheric CC background (expected events = {0:.2f})'
    #          .format(N_CCatmo_PSD), where='mid')
    # plt.step(E_visible, NCatmo_per_bin_PSD, color='orange',
    #          label='atmospheric NC background (expected events = {0:.2f})'.format(N_NCatmo_PSD), where='mid')
    # plt.step(E_visible, fast_neutron_per_bin_PSD, 'm-', label='fast neutron background (expected events = {0:.2f})'
    #          .format(N_fastneutron_PSD), where='mid')
    # plt.step(E_visible, spectrum_per_bin_PSD, 'k-', label='total spectrum', where='mid')
    # plt.xlim(E_vis_min, E_vis_max)
    # plt.ylim(ymin=0)
    # plt.xlabel("Visible energy in MeV")
    # plt.ylabel("Expected spectrum dN/dE in events/bin (bin-width = {0:.2f} MeV)".format(E_vis_bin))
    # plt.title("Expected spectrum in JUNO after {0:.0f} years for {1:d} MeV Dark Matter signal with PSD\n"
    #           "(IBD supp. = {2:.2f} %, NC supp. = {3:.2f} %, FN supp. = {4:.2f} %)"
    #           .format(t_years, dm_mass, IBD_supp, NC_supp, FN_supp))
    # plt.legend()
    # plt.grid()

    h3 = plt.figure(3, figsize=(15, 8))
    plt.semilogy(E_visible, signal_per_bin, 'r-', label='signal from DM annihilation for '
                 '$<\\sigma_Av>=${0:.1e}$cm^3/s$\nexpected events = {1:.2f}'.format(sigma_Anni, N_signal),
                 drawstyle="steps")
    plt.semilogy(E_visible, DSNB_per_bin, 'b-', label='DSNB background (expected events = {0:.2f})'
                 .format(N_DSNB), drawstyle="steps")
    plt.semilogy(E_visible, reactor_per_bin, 'c-', label='reactor background (expected events = {0:.2f})'
                 .format(N_reactor), drawstyle="steps")
    plt.semilogy(E_visible, CCatmo_per_bin, 'g-', label='atmospheric CC background (expected events = {0:.2f})'
                 .format(N_CCatmo), drawstyle="steps")
    plt.semilogy(E_visible, NCatmo_per_bin, color='orange', drawstyle="steps",
                 label='atmospheric NC background (expected events = {0:.2f})'.format(N_NCatmo))
    plt.semilogy(E_visible, fast_neutron_per_bin, 'm-', label='fast neutron background (expected events = {0:.2f})'
                 .format(N_fastneutron), drawstyle="steps")
    plt.semilogy(E_visible, spectrum_per_bin, 'k-', label='total spectrum', drawstyle="steps")
    plt.xlim(E_vis_min, E_vis_max)
    plt.ylim(10**(-2), 50)
    plt.xlabel("Visible energy in MeV")
    plt.ylabel("Expected spectrum dN/dE in events/bin (bin-width = {0:.2f} MeV)".format(E_vis_bin))
    plt.title("Expected spectrum in JUNO after {0:.0f} years for {1:d} MeV Dark Matter signal".format(t_years, dm_mass))
    # plt.title("Expected background spectrum in JUNO after {0:.0f} years".format(t_years))
    plt.legend()
    plt.grid()
    plt.close()

    h4 = plt.figure(4, figsize=(15, 8))
    plt.semilogy(E_visible, signal_per_bin_PSD, 'r-', drawstyle="steps", label='signal from DM annihilation for '
                 '$<\\sigma_Av>=${0:.1e}$cm^3/s$\nexpected events = {1:.2f}'.format(sigma_Anni, N_signal_PSD))
    plt.semilogy(E_visible, DSNB_per_bin_PSD, 'b-', label='DSNB background (expected events = {0:.2f})'
                 .format(N_DSNB_PSD), drawstyle="steps")
    plt.semilogy(E_visible, reactor_per_bin_PSD, 'c-', label='reactor background (expected events = {0:.2f})'
                 .format(N_reactor_PSD), drawstyle="steps")
    plt.semilogy(E_visible, CCatmo_per_bin_PSD, 'g-', label='atmospheric CC background (expected events = {0:.2f})'
                 .format(N_CCatmo_PSD), drawstyle="steps")
    plt.semilogy(E_visible, NCatmo_per_bin_PSD, color='orange', drawstyle="steps",
                 label='atmospheric NC background (expected events = {0:.2f})'.format(N_NCatmo_PSD))
    plt.semilogy(E_visible, fast_neutron_per_bin_PSD, 'm-', label='fast neutron background (expected events = {0:.2f})'
                 .format(N_fastneutron_PSD), drawstyle="steps")
    plt.semilogy(E_visible, spectrum_per_bin_PSD, 'k-', label='total spectrum', drawstyle="steps")
    plt.xlim(E_vis_min, E_vis_max)
    plt.ylim(10**(-2), 50)
    plt.xlabel("Visible energy in MeV")
    plt.ylabel("Expected spectrum dN/dE in events/bin (bin-width = {0:.2f} MeV)".format(E_vis_bin))
    plt.title("Expected spectrum in JUNO after {0:.0f} years for {1:d} MeV Dark Matter signal with PSD\n"
              "(IBD supp. = {2:.2f} %, NC supp. = 98.98 %, FN supp. = {3:.2f} %)"
              .format(t_years, dm_mass, IBD_supp, FN_supp))
    # plt.title("Expected background spectrum in JUNO after {0:.0f} years with PSD\n"
    #           "(IBD supp. = {1:.2f} %, NC supp. = 98.98 %, FN supp. = {2:.2f} %)"
    #           .format(t_years, IBD_supp, FN_supp))
    plt.legend()
    plt.grid()
    plt.savefig(path_spectra + "spectrum_{0:d}MeV_PSD.png".format(dm_mass))

    plt.close()
    # plt.show()
