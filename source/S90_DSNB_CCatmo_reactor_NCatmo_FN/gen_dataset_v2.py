""" Script gen_dataset_v2.py (29.10.2019):

    It is used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo_FN".

    The Script is a function, which generates datasets (virtual experiments) from the theoretical simulated spectra.

    The simulated spectra are generated with gen_spectrum_v3.py (S90_DSNB_CCatmo_reactor_NCatmo_FN).

    The script gen_dataset_v2.py is used in the scripts gen_dataset_v2_local.py (when you generate datasets on the
    local computer)
"""

# import of the necessary packages:
import datetime
import numpy as np
import matplotlib
# To generate images without having a window appear, do:
# The easiest way is use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# first define the function, which compares the 6 inputs:
def compare_6fileinputs(signal, dsnb, ccatmo, reactor, ncatmo, fastn, output_string):
    """
    function, which compares the input values of 6 files (signal, DSNB, CCatmo, reactor, NCatmo, FastN).
    If all six values are the same, the function returns the value of the signal-file.
    If one value differs from the other, the function prints a warning and returns a string.

    :param signal: value from the signal-file (float)
    :param dsnb: value from the DSNB-file (float)
    :param ccatmo: value from the CCatmo-file (float)
    :param reactor: value from the Reactor-file (float)
    :param ncatmo: value from the NCatmo file (float)
    :param fastn: value from the fast Neutron file (float)
    :param output_string: string variable, which describes the value (e.g. 'interval_E_visible') (string)
    :return: output: either the value of the signal-file (float) or the string output_string (string)
    """

    if signal == dsnb == ccatmo == reactor == ncatmo == fastn:
        output = signal
    else:
        output = output_string
        print("ERROR: variable {0} is not the same for the different files!".format(output))

    return output


# define the function, which generates the datasets:
def gen_dataset_v2(mass_dm, save_data_txt, save_data_all, display_data, dataset_start, dataset_stop, path_output,
                   path_dataset, file_signal, file_signal_info, file_dsnb, file_dsnb_info, file_ccatmo,
                   file_ccatmo_info, file_reactor, file_reactor_info, file_ncatmo, file_ncatmo_info, file_fastn,
                   file_fastn_info):
    """
    Function to generate datasets (virtual experiments).
    :param mass_dm: Dark matter mass in MeV (float)
    :param save_data_txt: boolean variable, which controls, if the txt-file with the data are saved
    (save_data_txt=True) or not (save_data = False) (Boolean)
    :param save_data_all: boolean variable, which controls, if the all files (txt and png) with the data are saved
    (save_data_png=True) or not (save_data_png = False) (Boolean)
    :param display_data: boolean variable, which controls, if the data is displayed on screen (display_data = True)
    or not (display_data = False) (Boolean)
    :param dataset_start: defines the start point (integer)
    :param dataset_stop: defines the end point (integer)
    :param path_output: path of the dataset_output folder, e.g. 'path_folder + "dataset_output_{DM_mass}"' (string)
    :param path_dataset: path of the folder, where the datasets are saved, e.g. 'path_output + "/datasets"' (string)
    :param file_signal: file name of the simulated signal spectrum with PSD (string)
    :param file_signal_info: file name of the information about the signal spectrum with PSD (string)
    :param file_dsnb: file name of the simulated DSNB background spectrum with PSD (string)
    :param file_dsnb_info: file name of the information about the DSNB background spectrum with PSD (string)
    :param file_ccatmo: file name of the simulated CCatmo background spectrum with PSD (string)
    :param file_ccatmo_info: file name of the information about the simulated CCatmo background spectrum with PSD
                            (string)
    :param file_reactor: file name of the simulated reactor background spectrum with PSD (string)
    :param file_reactor_info: file name of the information about the simulated reactor background spectrum with PSD
                                (string)
    :param file_ncatmo: file name of the simulated NCatmo background spectrum with PSD (string)
    :param file_ncatmo_info: file name of the information about the simulated NCatmo background spectrum with PSD
                            (string)
    :param file_fastn: file name of the simulated fast neutron background spectrum with PSD (string)
    :param file_fastn_info: file name of the information about the simulated fast neutron background spectrum with PSD
                        (string)

    :return:
    """

    """ Variable, which defines the date and time of running the script: """
    # get the date and time, when the script was run:
    date = datetime.datetime.now()
    now = date.strftime("%Y-%m-%d %H:%M")

    """ Load the calculated spectrum of the signal from the txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_signal = np.loadtxt(file_signal)
    # load information about the above file (np.array of float):
    info_signal = np.loadtxt(file_signal_info)
    # get bin-width of E_neutrino in MeV from info-file (float):
    interval_e_neutrino_signal = info_signal[2]
    # get minimum of E_neutrino in MeV from info-file (float):
    min_e_neutrino_signal = info_signal[0]
    # get maximum of E_neutrino in MeV from info-file (float):
    max_e_neutrino_signal = info_signal[1]
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_signal = info_signal[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_signal = info_signal[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_signal = info_signal[4]
    # get exposure time in years from info-file (float):
    t_years_signal = info_signal[6]
    # get number of free protons in the detector from info-file(float):
    n_target_signal = info_signal[7]
    # get detection efficiency for Inverse Beta Decay from info-file (float):
    det_eff_signal = info_signal[8]
    # get PSD suppression for IBD events from info-file (float):
    psd_suppression_ibd = info_signal[9]
    # get DM mass in MeV from info-file (float):
    mass_darkmatter = info_signal[10]
    # get number of neutrinos from the spectrum from info-file (float):
    n_neutrino_signal = info_signal[12]
    # get number of neutrinos from spectrum after PSD (float):
    n_neutrino_signal_after_psd = info_signal[13]
    # get DM annihilation cross-section in cm**3/s from info-file (float):
    sigma_annihilation = info_signal[14]
    # get angular-averaged intensity over whole Milky Way from info.file (float):
    j_avg = info_signal[15]
    # get electron-antineutrino flux at Earth in 1/(MeV*s*cm**2) from info-file (float):
    flux_signal = info_signal[16]
    # get the exposure ratio of muon veto cut (float):
    exposure_ratio_muon_veto = info_signal[17]

    """ Check if mass_dm from the input is equal to mass_darkmatter from the information file: """
    if mass_dm != mass_darkmatter:
        print("Error: DM mass from input is NOT equal to DM mass from info-file")

    """ Load the calculated spectrum of the DSNB background from the txt-file: """
    # load calculated spectrum from corresponding file (events/MeV) (np.array of float):
    spectrum_dsnb = np.loadtxt(file_dsnb)
    # load information about the above file (np.array of float):
    info_dsnb = np.loadtxt(file_dsnb_info)
    # get bin-width of E_neutrino in MeV from info-file (float):
    interval_e_neutrino_dsnb = info_dsnb[2]
    # get minimum of E_neutrino in MeV from info-file (float):
    min_e_neutrino_dsnb = info_dsnb[0]
    # get maximum of E_neutrino in MeV from info-file (float):
    max_e_neutrino_dsnb = info_dsnb[1]
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_dsnb = info_dsnb[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_dsnb = info_dsnb[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_dsnb = info_dsnb[4]
    # get exposure time in years from info-file (float):
    t_years_dsnb = info_dsnb[6]
    # get number of free protons in the detector from info-file(float):
    n_target_dsnb = info_dsnb[7]
    # get detection efficiency for Inverse Beta Decay from info-file (float):
    det_eff_dsnb = info_dsnb[8]
    # get PSD suppression for IBD events from info-file (float):
    psd_suppression_ibd = info_dsnb[9]
    # get number of neutrinos from the spectrum from info-file (float):
    n_neutrino_dsnb = info_dsnb[11]
    # get number of neutrinos from the spectrum after PSD from info-file (float):
    n_neutrino_dsnb_after_psd = info_dsnb[12]
    # get mean energy of nu_Ebar in MeV from info-file (float):
    e_mean_nu_e_bar = info_dsnb[13]
    # get pinching factor for nu_Ebar from info-file (float):
    beta_nu_e_bar = info_dsnb[14]
    # get mean energy of nu_Xbar in MeV from info-file (float):
    e_mean_nu_x_bar = info_dsnb[15]
    # get pinching factor for nu_Xbar from info-file (float):
    beta_nu_x_bar = info_dsnb[16]
    # get correction factor of SFR from info-file (float):
    f_star = info_dsnb[17]
    # get the exposure ratio of muon veto cut (float):
    exposure_ratio_muon_veto = info_dsnb[18]

    """ load the calculated spectrum of the atmospheric CC background from txt-file: """
    # load calculated spectrum from corresponding file (events/MeV) (np.array of float):
    spectrum_ccatmo = np.loadtxt(file_ccatmo)
    # load information about the above file (np.array of float):
    info_ccatmo = np.loadtxt(file_ccatmo_info)
    # get bin-width of E_neutrino in MeV from info-file (float):
    interval_e_neutrino_ccatmo = info_ccatmo[2]
    # get minimum of E_neutrino in MeV from info-file (float):
    min_e_neutrino_ccatmo = info_ccatmo[0]
    # get maximum of E_neutrino in MeV from info-file (float):
    max_e_neutrino_ccatmo = info_ccatmo[1]
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_ccatmo = info_ccatmo[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_ccatmo = info_ccatmo[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_ccatmo = info_ccatmo[4]
    # get exposure time in years from info-file (float):
    t_years_ccatmo = info_ccatmo[6]
    # get number of free protons in the detector from info-file(float):
    n_target_ccatmo = info_ccatmo[7]
    # get detection efficiency for Inverse Beta Decay from info-file (float):
    det_eff_ccatmo = info_ccatmo[8]
    # get PSD suppression for IBD events from info-file (float):
    psd_suppression_ibd = info_ccatmo[9]
    # get number of neutrinos from the spectrum from info-file (float):
    n_neutrino_ccatmo = info_ccatmo[11]
    # get number of neutrinos from the spectrum after PSD from info-file (float):
    n_neutrino_ccatmo_after_psd = info_ccatmo[12]
    # get info if oscillation is considered (0=no, 1=yes) from info-file (float):
    oscillation = info_ccatmo[13]
    # get survival probability of nu_Ebar (P_e_to_e) from info-file (float):
    p_ee = info_ccatmo[14]
    # get oscillation probability of nu_Mubar to nu_Ebar (P_mu_to_e) from info-file (float):
    p_mue = info_ccatmo[15]
    # get the exposure ratio of muon veto cut (float):
    exposure_ratio_muon_veto = info_ccatmo[16]

    """ load the calculated spectrum of the reactor background from txt-file: """
    # load calculated spectrum from corresponding file (events/MeV) (np.array of float):
    spectrum_reactor = np.loadtxt(file_reactor)
    # load information about the above file (np.array of float):
    info_reactor = np.loadtxt(file_reactor_info)
    # get bin-width of E_neutrino in MeV from info-file (float):
    interval_e_neutrino_reactor = info_reactor[2]
    # get minimum of E_neutrino in MeV from info-file (float):
    min_e_neutrino_reactor = info_reactor[0]
    # get maximum of E_neutrino in MeV from info-file (float):
    max_e_neutrino_reactor = info_reactor[1]
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_reactor = info_reactor[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_reactor = info_reactor[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_reactor = info_reactor[4]
    # get exposure time in years from info-file (float):
    t_years_reactor = info_reactor[6]
    # get number of free protons in the detector from info-file(float):
    n_target_reactor = info_reactor[7]
    # get detection efficiency for Inverse Beta Decay from info-file (float):
    det_eff_reactor = info_reactor[8]
    # get PSD suppression for IBD events from info-file (float):
    psd_suppression_ibd = info_reactor[9]
    # get number of neutrinos from the spectrum from info-file (float):
    n_neutrino_reactor = info_reactor[11]
    # get number of neutrinos from the spectrum after PSD from info-file (float):
    n_neutrino_reactor_after_psd = info_reactor[12]
    # get thermal power of NPP in GW from info-file (float):
    power_thermal = info_reactor[13]
    # get fission fraction of U235 from info-file (float):
    fraction_u235 = info_reactor[14]
    # get fission fraction of U238 from info-file (float):
    fraction_u238 = info_reactor[15]
    # get fission fraction of Pu239 from info-file (float):
    fraction_pu239 = info_reactor[16]
    # get fission fraction of Pu241 from info-file (float):
    fraction_pu241 = info_reactor[17]
    # get distance from reactor to detector in meter from info-file (float):
    l_meter = info_reactor[18]
    # get the exposure ratio of muon veto cut (float):
    exposure_ratio_muon_veto = info_reactor[19]

    """ Load the calculated spectrum of the NCatmo background from the txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_ncatmo = np.loadtxt(file_ncatmo)
    # load information about the above file (np.array of float):
    info_ncatmo = np.loadtxt(file_ncatmo_info)
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_ncatmo = info_ncatmo[2]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_ncatmo = info_ncatmo[0]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_ncatmo = info_ncatmo[1]
    # get exposure time in years from info-file (float):
    t_years_ncatmo = info_ncatmo[3]
    # get the radius in meter of the applied volume cut (float):
    radius_vol_cut_ncatmo = info_ncatmo[4]
    # get total number of simulated NC events on C12 (float):
    number_simu_total_ncatmo = info_ncatmo[5]
    # get number of IBD-like NC events in spectrum JUNO will measure wo PSD (float):
    number_events_wo_psd_ncatmo = info_ncatmo[6]
    # get number of IBD-like events after PSD (float):
    number_events_w_psd_ncatmo = info_ncatmo[7]
    # theoretical NC event rate in JUNO detector in NC events/sec (float):
    event_rate_ncatmo = info_ncatmo[8]

    """ Load the calculated spectrum of the fast neutron background from the txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_fastn = np.loadtxt(file_fastn)
    # load information about the above file (np.array of float):
    info_fastn = np.loadtxt(file_fastn_info)
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_fastn = info_fastn[2]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_fastn = info_fastn[0]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_fastn = info_fastn[1]
    # get exposure time in years from info-file (float):
    t_years_fastn = info_fastn[3]
    # get the radius in meter of the applied volume cut (float):
    radius_vol_cut_fastn = info_fastn[4]
    # get number of events from info-file (float):
    n_events_fastn = info_fastn[6]
    # get PSD suppression of fast neutrons (float):
    psd_suppression_fastn = info_fastn[7]
    # get number of events after PSD from info-file (float):
    n_events_fastn_after_psd = info_fastn[8]
    # get the exposure ratio of muon veto cut (float):
    exposure_ratio_muon_veto = info_fastn[9]

    """ Check, if the 'input'-parameter of the files are the same (are the spectra generated with the 
    same properties?): """
    # check interval_e_visible:
    interval_e_visible = compare_6fileinputs(interval_e_visible_signal, interval_e_visible_dsnb,
                                             interval_e_visible_ccatmo, interval_e_visible_reactor,
                                             interval_e_visible_ncatmo, interval_e_visible_fastn,
                                             'interval_e_visible')
    # check min_e_visible:
    min_e_visible = compare_6fileinputs(min_e_visible_signal, min_e_visible_dsnb, min_e_visible_ccatmo,
                                        min_e_visible_reactor, min_e_visible_ncatmo, min_e_visible_fastn,
                                        'min_e_visible')
    # check max_e_visible:
    max_e_visible = compare_6fileinputs(max_e_visible_signal, max_e_visible_dsnb, max_e_visible_ccatmo,
                                        max_e_visible_reactor, max_e_visible_ncatmo, max_e_visible_fastn,
                                        'max_e_visible')
    # check t_years (exposure time in years):
    t_years = compare_6fileinputs(t_years_signal, t_years_dsnb, t_years_ccatmo, t_years_reactor, t_years_ncatmo,
                                  t_years_fastn, 't_years')

    """ define the visible energy in MeV (arange till max + interval to get an array from min to max) 
        (np.array of float): """
    e_visible = np.arange(min_e_visible, max_e_visible + interval_e_visible, interval_e_visible)

    """ Total simulated spectrum for JUNO after t_years years and for DM with mass mass_DM in 1/bin 
    (np.array of float): """
    spectrum_signal_per_bin = spectrum_signal
    spectrum_dsnb_per_bin = spectrum_dsnb * interval_e_visible
    spectrum_ccatmo_per_bin = spectrum_ccatmo * interval_e_visible
    spectrum_reactor_per_bin = spectrum_reactor * interval_e_visible
    spectrum_ncatmo_per_bin = spectrum_ncatmo
    spectrum_fastn_per_bin = spectrum_fastn

    # Number of events per bin (spectrum per bin) of e_visible (np.array of float):
    spectrum_total_per_bin = (spectrum_signal_per_bin + spectrum_dsnb_per_bin +
                              spectrum_ccatmo_per_bin + spectrum_reactor_per_bin + spectrum_ncatmo_per_bin +
                              spectrum_fastn_per_bin)

    """ Generate datasets (virtual experiments) from the total simulated spectrum: """
    # iteration of index has to go to number_dataset_stop + 1, because np.arange(start, stop)
    # generate values in the half-open interval [start, stop) -> stop would be excluded
    for index in np.arange(dataset_start, dataset_stop + 1):
        # spectrum of the dataset:
        # generate for each entry in spectrum_total_per_bin a poisson-distributed integer-value, which describes the
        # number of events a virtual experiment would measure in the corresponding bin (units: number of events
        # per bin) (np.array of integer):
        spectrum_dataset_per_bin = np.random.poisson(spectrum_total_per_bin)

        # to save the txt-files of the dataset values, save_data_txt must be True:
        if save_data_txt:
            # save Spectrum_dataset to txt-dataset-file:
            np.savetxt(path_dataset + '/Dataset_{0:d}.txt'.format(index), spectrum_dataset_per_bin, fmt='%4.5f',
                       header='Spectrum of virtual experiment in number of events per bin'
                              '(Dataset generated with gen_dataset_v2.py, {0}):\n'
                              'Input files:\n{1},\n{2},\n{3},\n{4},\n{5},\n{6}:'
                       .format(now, file_signal, file_dsnb, file_ccatmo, file_reactor, file_ncatmo, file_fastn))

        # to save the png-figures of the dataset, save_data_all must be True:
        if save_data_all:
            # build a step-plot of the generated dataset and save the image to png-file:
            h1 = plt.figure(1)
            plt.step(e_visible, spectrum_dataset_per_bin, where='mid')
            plt.xlabel('Visible energy in MeV')
            plt.ylabel('Number of events per bin per {0:.0f}yr (bin={1:.2f}MeV)'.format(t_years, interval_e_visible))
            plt.title('Spectrum that JUNO will measure after {0:.0f} years for DM of mass = {1:.0f} MeV'
                      .format(t_years, mass_dm))
            plt.ylim(ymin=0)
            plt.xticks(np.arange(min_e_visible, max_e_visible, 10))
            plt.grid()
            plt.savefig(path_dataset + '/Dataset_{0:d}.png'.format(index))
            plt.close(h1)

    # to save information about the dataset, save_data_txt must be True:
    if save_data_txt:
        # save information about dataset to info-dataset-file:
        np.savetxt(path_dataset + '/info_dataset_{0:d}_to_{1:d}.txt'.format(dataset_start, dataset_stop),
                   np.array([interval_e_visible, min_e_visible, max_e_visible, interval_e_neutrino_ccatmo,
                             min_e_neutrino_ccatmo, max_e_neutrino_ccatmo, t_years, n_target_ccatmo, det_eff_ccatmo,
                             psd_suppression_ibd, psd_suppression_fastn]),
                   fmt='%1.9e',
                   header='Information about the dataset-simulation Dataset_{0:d}.txt to Dataset_{1:d}.txt:\n'
                          'Input files:\n{2},\n{3},\n{4},\n{5},\n{6},\n{7}\n'
                          'bin-width E_visible in MeV, minimum E_visible in MeV, maximum E_visible in MeV,\n'
                          'bin-width E_neutrino in MeV, minimum E_neutrino in MeV, maximum E_neutrino in MeV\n'
                          'exposure time in years, number of targets (free protons), IBD detection efficiency,\n'
                          'PSD suppression of IBD events, PSD suppression of fast neutron events.'
                   .format(dataset_start, dataset_stop, file_signal, file_dsnb, file_ccatmo, file_reactor, file_ncatmo,
                           file_fastn))

    # to save the simulated total spectrum, save_data_all must be True:
    if save_data_all:
        # save total simulated spectrum to spectrum-file:
        np.savetxt(path_output + '/spectrum_simulated.txt', spectrum_total_per_bin, fmt='%4.5f',
                   header='Total simulated spectrum in events/bin:\n'
                          '(sum of single spectra from input files:\n'
                          '{0},\n{1},\n{2},\n{3},\n{4},\n{5})'.format(file_signal, file_dsnb, file_ccatmo, file_reactor,
                                                                      file_ncatmo, file_fastn))

    # Total simulated spectrum and one dataset is displayed, when DISPLAY_SPECTRA is True:
    if display_data:
        # Display the total simulated spectrum and with different backgrounds:
        h2 = plt.figure(2)
        plt.step(e_visible, spectrum_total_per_bin, where='mid', label='total spectrum')
        plt.step(e_visible, spectrum_signal_per_bin, '--', where='mid',
                 label='DM signal ($<\sigma_Av>=${0:.1e}$cm^3/s$)'
                 .format(sigma_annihilation))
        plt.step(e_visible, spectrum_dsnb_per_bin, '--', where='mid', label='DSNB background')
        plt.step(e_visible, spectrum_ccatmo_per_bin, '--', where='mid', label='atmospheric CC background')
        plt.step(e_visible, spectrum_reactor_per_bin, '--', where='mid', label='reactor background')
        plt.step(e_visible, spectrum_ncatmo_per_bin, '--', where='mid', label='atmospheric NC background')
        plt.step(e_visible, spectrum_fastn_per_bin, '--', where='mid', label='fast neutron background')
        plt.xlabel('Visible energy in MeV')
        plt.ylabel(
            'Simulated spectrum in events/(bin*{0:.0f}yr) (bin={1:.2f}MeV)'.format(t_years, interval_e_visible))
        plt.title(
            'Simulated spectrum in JUNO after {0:.0f} years for DM of mass = {1:.0f} MeV'.format(t_years, mass_dm))
        plt.ylim(ymin=0)
        plt.xticks(np.arange(min_e_visible, max_e_visible, 10))
        plt.legend()
        plt.grid()

        # Display the total simulated spectrum and with different backgrounds (in LOGARITHMIC scale):
        h3 = plt.figure(3)
        plt.semilogy(e_visible, spectrum_total_per_bin, label='total spectrum', drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_signal_per_bin, '--', label='DM signal ($<\sigma_Av>=${0:.1e}$cm^3/s$)'
                     .format(sigma_annihilation), drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_dsnb_per_bin, '--', label='DSNB background', drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_ccatmo_per_bin, '--', label='atmospheric CC background',
                     drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_reactor_per_bin, '--', label='reactor background', drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_ncatmo_per_bin, '--', label='atmospheric NC background', drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_fastn_per_bin, '--', label='fast neutron background', drawstyle='steps-mid')
        plt.xlabel('Visible energy in MeV')
        plt.ylabel(
            'Simulated spectrum in events/(bin*{0:.0f}yr) (bin={1:.2f}MeV)'.format(t_years, interval_e_visible))
        plt.title(
            'Simulated spectrum in JUNO after {0:.0f} years for DM of mass = {1:.0f} MeV'.format(t_years, mass_dm))
        plt.ylim(ymin=0.01, ymax=10)
        plt.xticks(np.arange(min_e_visible, max_e_visible, 10))
        plt.legend()
        plt.grid()

        plt.show()

    return

