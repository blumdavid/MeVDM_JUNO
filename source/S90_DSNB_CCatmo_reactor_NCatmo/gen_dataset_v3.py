""" Script gen_dataset_v3.py (12.05.2020):

    It is used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo".

    The Script is a function, which generates datasets (virtual experiments) from the theoretical simulated spectra.

    The simulated spectra are generated with gen_spectrum_v4.py (S90_DSNB_CCatmo_reactor_NCatmo).

    The script gen_dataset_v3.py is used in the scripts gen_dataset_v3_local.py (when you generate datasets on the
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
def compare_6fileinputs(signal, dsnb, ccatmo_p, reactor, ncatmo, ccatmo_c12, output_string):
    """
    function, which compares the input values of 6 files (signal, DSNB, CCatmo, reactor, NCatmo, FastN).
    If all six values are the same, the function returns the value of the signal-file.
    If one value differs from the other, the function prints a warning and returns a string.

    :param signal: value from the signal-file (float)
    :param dsnb: value from the DSNB-file (float)
    :param ccatmo_p : value from the CCatmo on p-file (float)
    :param reactor: value from the Reactor-file (float)
    :param ncatmo: value from the NCatmo file (float)
    :param ccatmo_c12: value from the CCatmo on C12 file (float)
    :param output_string: string variable, which describes the value (e.g. 'interval_E_visible') (string)
    :return: output: either the value of the signal-file (float) or the string output_string (string)
    """

    if signal == dsnb == ccatmo_p == reactor == ncatmo == ccatmo_c12:
        output = signal
    else:
        output = output_string
        print("ERROR: variable {0} is not the same for the different files!".format(output))

    return output


# define the function, which generates the datasets:
def gen_dataset_v3(mass_dm, save_data_txt, save_data_all, display_data, dataset_start, dataset_stop, path_output,
                   path_dataset, file_signal, file_signal_info, file_dsnb, file_dsnb_info, file_ccatmo_p,
                   file_ccatmo_p_info, file_reactor, file_reactor_info, file_ncatmo, file_ncatmo_info, file_ccatmo_c12,
                   file_ccatmo_c12_info):
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
    :param file_ccatmo_p: file name of the simulated CCatmo background spectrum on p with PSD (string)
    :param file_ccatmo_p_info: file name of the information about the simulated CCatmo background spectrum on p with PSD
                            (string)
    :param file_reactor: file name of the simulated reactor background spectrum with PSD (string)
    :param file_reactor_info: file name of the information about the simulated reactor background spectrum with PSD
                                (string)
    :param file_ncatmo: file name of the simulated NCatmo background spectrum with PSD (string)
    :param file_ncatmo_info: file name of the information about the simulated NCatmo background spectrum with PSD
                            (string)
    :param file_ccatmo_c12: file name of the simulated fast neutron background spectrum on C12 with PSD (string)
    :param file_ccatmo_c12_info: file name of the information about the simulated fast neutron background spectrum
                            on C12with PSD (string)

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
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_signal = info_signal[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_signal = info_signal[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_signal = info_signal[4]
    # get exposure time in years from info-file (float):
    t_years_signal = info_signal[6]
    # get DM mass in MeV from info-file (float):
    mass_darkmatter = info_signal[10]
    # get DM annihilation cross-section in cm**3/s from info-file (float):
    sigma_annihilation = info_signal[14]

    """ Check if mass_dm from the input is equal to mass_darkmatter from the information file: """
    if mass_dm != mass_darkmatter:
        print("Error: DM mass from input is NOT equal to DM mass from info-file")

    """ Load the calculated spectrum of the DSNB background from the txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_dsnb = np.loadtxt(file_dsnb)
    # load information about the above file (np.array of float):
    info_dsnb = np.loadtxt(file_dsnb_info)
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_dsnb = info_dsnb[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_dsnb = info_dsnb[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_dsnb = info_dsnb[4]
    # get exposure time in years from info-file (float):
    t_years_dsnb = info_dsnb[6]

    """ load the calculated spectrum of the atmospheric CC background on protons from txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_ccatmo_p = np.loadtxt(file_ccatmo_p)
    # load information about the above file (np.array of float):
    info_ccatmo_p = np.loadtxt(file_ccatmo_p_info)
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_ccatmo_p = info_ccatmo_p[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_ccatmo_p = info_ccatmo_p[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_ccatmo_p = info_ccatmo_p[4]
    # get exposure time in years from info-file (float):
    t_years_ccatmo_p = info_ccatmo_p[6]

    """ load the calculated spectrum of the reactor background from txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_reactor = np.loadtxt(file_reactor)
    # load information about the above file (np.array of float):
    info_reactor = np.loadtxt(file_reactor_info)
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_reactor = info_reactor[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_reactor = info_reactor[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_reactor = info_reactor[4]
    # get exposure time in years from info-file (float):
    t_years_reactor = info_reactor[6]

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

    """ Load the calculated spectrum of the fast neutron background from the txt-file: """
    # load calculated spectrum from corresponding file (events/bin) (np.array of float):
    spectrum_ccatmo_c12 = np.loadtxt(file_ccatmo_c12)
    # load information about the above file (np.array of float):
    info_ccatmo_c12 = np.loadtxt(file_ccatmo_c12_info)
    # get bin-width of e_visible in MeV from info-file (float):
    interval_e_visible_ccatmo_c12 = info_ccatmo_c12[5]
    # get minimum of e_visible in MeV from info-file (float):
    min_e_visible_ccatmo_c12 = info_ccatmo_c12[3]
    # get maximum of e_visible in MeV from info-file (float):
    max_e_visible_ccatmo_c12 = info_ccatmo_c12[4]
    # get exposure time in years from info-file (float):
    t_years_ccatmo_c12 = info_ccatmo_c12[6]

    """ Check, if the 'input'-parameter of the files are the same (are the spectra generated with the 
    same properties?): """
    # check interval_e_visible:
    interval_e_visible = compare_6fileinputs(interval_e_visible_signal, interval_e_visible_dsnb,
                                             interval_e_visible_ccatmo_p, interval_e_visible_reactor,
                                             interval_e_visible_ncatmo, interval_e_visible_ccatmo_c12,
                                             'interval_e_visible')
    # check min_e_visible:
    min_e_visible = compare_6fileinputs(min_e_visible_signal, min_e_visible_dsnb, min_e_visible_ccatmo_p,
                                        min_e_visible_reactor, min_e_visible_ncatmo, min_e_visible_ccatmo_c12,
                                        'min_e_visible')
    # check max_e_visible:
    max_e_visible = compare_6fileinputs(max_e_visible_signal, max_e_visible_dsnb, max_e_visible_ccatmo_p,
                                        max_e_visible_reactor, max_e_visible_ncatmo, max_e_visible_ccatmo_c12,
                                        'max_e_visible')
    # check t_years (exposure time in years):
    t_years = compare_6fileinputs(t_years_signal, t_years_dsnb, t_years_ccatmo_p, t_years_reactor, t_years_ncatmo,
                                  t_years_ccatmo_c12, 't_years')

    """ define the visible energy in MeV (arange till max + interval to get an array from min to max) 
        (np.array of float): """
    e_visible = np.arange(min_e_visible, max_e_visible + interval_e_visible, interval_e_visible)

    """ Total simulated spectrum for JUNO after t_years years and for DM with mass mass_DM in 1/bin 
    (np.array of float): """
    spectrum_signal_per_bin = spectrum_signal
    spectrum_dsnb_per_bin = spectrum_dsnb
    spectrum_ccatmo_p_per_bin = spectrum_ccatmo_p
    spectrum_reactor_per_bin = spectrum_reactor
    spectrum_ncatmo_per_bin = spectrum_ncatmo
    spectrum_ccatmo_c12_per_bin = spectrum_ccatmo_c12

    # Number of events per bin (spectrum per bin) of e_visible (np.array of float):
    spectrum_total_per_bin = (spectrum_signal_per_bin + spectrum_dsnb_per_bin +
                              spectrum_ccatmo_p_per_bin + spectrum_reactor_per_bin + spectrum_ncatmo_per_bin +
                              spectrum_ccatmo_c12_per_bin)

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
                       .format(now, file_signal, file_dsnb, file_ccatmo_p, file_reactor, file_ncatmo, file_ccatmo_c12))

        # to save the png-figures of the dataset, save_data_all must be True:
        if save_data_all:
            # build a step-plot of the generated dataset and save the image to png-file:
            h1 = plt.figure(1)
            plt.step(e_visible, spectrum_dataset_per_bin, where='mid')
            plt.xlabel('Visible energy in MeV')
            plt.ylabel('Number of events per bin per {0:.0f}yr (bin={1:.2f}MeV)'.format(t_years, interval_e_visible))
            plt.title('Spectrum that JUNO will measure after {0:.0f} years (background only)'
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
                   np.array([interval_e_visible, min_e_visible, max_e_visible, t_years]),
                   fmt='%1.9e',
                   header='Information about the dataset-simulation Dataset_{0:d}.txt to Dataset_{1:d}.txt:\n'
                          'Input files:\n{2},\n{3},\n{4},\n{5},\n{6},\n{7}\n'
                          'bin-width E_visible in MeV, minimum E_visible in MeV, maximum E_visible in MeV,\n'
                          'exposure time in years:.'
                   .format(dataset_start, dataset_stop, file_signal, file_dsnb, file_ccatmo_p, file_reactor, file_ncatmo,
                           file_ccatmo_c12))

    # to save the simulated total spectrum, save_data_all must be True:
    if save_data_all:
        # save total simulated spectrum to spectrum-file:
        np.savetxt(path_output + '/spectrum_simulated.txt', spectrum_total_per_bin, fmt='%4.5f',
                   header='Total simulated spectrum in events/bin:\n'
                          '(sum of single spectra from input files:\n'
                          '{0},\n{1},\n{2},\n{3},\n{4},\n{5})'.format(file_signal, file_dsnb, file_ccatmo_p,
                                                                      file_reactor,
                                                                      file_ncatmo, file_ccatmo_c12))

    # Total simulated spectrum and one dataset is displayed, when DISPLAY_SPECTRA is True:
    if display_data:
        # Display the total simulated spectrum and with different backgrounds:
        h2 = plt.figure(2)
        plt.step(e_visible, spectrum_total_per_bin, where='mid', label='total spectrum')
        plt.step(e_visible, spectrum_signal_per_bin, '--', where='mid',
                 label='DM signal ($<\sigma_Av>=${0:.1e}$cm^3/s$)'
                 .format(sigma_annihilation))
        plt.step(e_visible, spectrum_dsnb_per_bin, '--', where='mid', label='DSNB background')
        plt.step(e_visible, spectrum_ccatmo_p_per_bin, '--', where='mid', label='atmospheric CC background on protons')
        plt.step(e_visible, spectrum_reactor_per_bin, '--', where='mid', label='reactor background')
        plt.step(e_visible, spectrum_ncatmo_per_bin, '--', where='mid', label='atmospheric NC background')
        plt.step(e_visible, spectrum_ccatmo_c12_per_bin, '--', where='mid', label='atmospheric CC background on C12')
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
        plt.semilogy(e_visible, spectrum_ccatmo_p_per_bin, '--', label='atmospheric CC background on protons',
                     drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_reactor_per_bin, '--', label='reactor background', drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_ncatmo_per_bin, '--', label='atmospheric NC background', drawstyle='steps-mid')
        plt.semilogy(e_visible, spectrum_ccatmo_c12_per_bin, '--', label='atmospheric CC background on C12',
                     drawstyle='steps-mid')
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

