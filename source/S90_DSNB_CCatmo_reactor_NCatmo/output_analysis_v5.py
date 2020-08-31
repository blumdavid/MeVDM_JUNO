""" Script "output_analysis_v5.py" from 12.05.2020.
    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo".

    output_analysis_v5.py:

    The Script is a function to display and analyze the results of the MCMC analysis.

    The MCMC analysis is done either with analyze_spectra_v7_local.py or with analyze_spectra_v7_server2.py.

    The script output_analysis_v5.py is used in the script output_analysis_v4_server.py
    (when you display the results on the server)
"""

import numpy as np
import datetime


# define the function, which analyzes the output of the analysis:
def output_analysis(save_data, dm_mass, path_output, path_analysis, file_info_analysis, file_number_start,
                    file_number_stop):
    """
    :param save_data: boolean value to define, if the result of output_analysis_v3.py is saved (boolean)
    :param dm_mass: DM mass in MeV (float)
    :param path_output: path of the output folder (output folder is "dataset_output_{}") (string)
    :param path_analysis: the path of the analysis folder (analysis folder is "analysis_mcmc") (string)
    :param file_info_analysis: path of the file, where the information about the analysis is saved (string)
    :param file_number_start: number of the files, that should be read in (is equal to dataset_start and dataset_stop)
    (integer)
    :param file_number_stop: number of the files, that should be read in (is equal to dataset_start and dataset_stop)
    (integer)

    :return:
    number_of_entries: number of files, that are read in (Equivalent to the number of entries in the result array)
    (integer)
    lower_energy_bound: lower bound of the energy window in MeV (float)
    upper_energy_bound: upper bound of the energy window in MeV (float)
    s_mode: mode of the total number of signal events (np.array of float)
    s_50: mean of the array s_mode (float)
    s_50_sigma: standard deviation of the array s_mode (float)
    s_50_16: 16% confidence level of the array s_mode (float)
    s_50_84: 84% confidence level of the array s_mode (float)
    s_50_2_5: 2.5% confidence level of the array s_mode (float)
    s_50_97_5: 97.5% confidence level of the array s_mode (float)
    s_50_0_15: 0.15% confidence level of the array s_mode (float)
    s_50_99_85: 99.85% confidence level of the array s_mode (float)
    signal_expected: number of expected signal events in the energy window (float)
    s_90_limit: values of upper 90 percent limit of the number of signal events (np.array of float)
    s_90: mean of s_90_limit (float)
    s_90_sigma: std of s_90_limit (float)
    s_90_16: 16% confidence level of s_90_limit (float)
    s_90_84: 84% confidence level of s_90_limit (float)
    s_90_2_5: 2.5% confidence level of s_90_limit (float)
    s_90_97.5: 97.5% confidence level of s_90_limit (float)
    s_90_0_15: 0.15% confidence level of s_90_limit (float)
    s_90_99_85: 99.85% confidence level of s_90_limit (float)
    dsnb_mode: mode of the total number of DSNB background events (np.array of float)
    dsnb_50: mean of dsnb_mode (float)
    dsnb_50_sigma: std of dsnb_mode (float)
    dsnb_50_16: 16% confidence level of dsnb_mode (float)
    dsnb_50_84: 84% confidence level of dsnb_mode (float)
    dsnb_expected: number of expected DSNB background events in the energy window (float)
    ccatmo_p_mode: mode of the total number of CCatmo background events on p (np.array of float)
    ccatmo_p_50: mean of ccatmo_mode (float)
    ccatmo_p_50_sigma: std of ccatmo_mode (float)
    ccatmo_p_50_16: 16% C.L. of ccatmo_mode (float)
    ccatmo_p_50_84: 84% C.L. of ccatmo_mode (float)
    ccatmo_p_expected: number of expected atmospheric CC background events on p in the energy window (float)
    reactor_mode: mode of the total number of reactor background events (np.array of float)
    reactor_50: mean of reactor_mode (float)
    reactor_50_sigma: std of reactor_mode (float)
    reactor_50_16: 16% C.L. of reactor_mode (float)
    reactor_50_84: 84% C.L. of reactor_mode (float)
    reactor_expected: number of expected reactor background events in the energy window (float)
    ncatmo_mode: mode of the total number of NCatmo background events (np.array of float)
    ncatmo_50: mean of ncatmo_mode (float)
    ncatmo_50_sigma: std of ncatmo_mode (float)
    ncatmo_50_16: 16% C.L. of ncatmo_mode (float)
    ncatmo_50_84: 84% C.L. of ncatmo_mode (float)
    ncatmo_expected: number of expected atmospheric NC background events in the energy window (float)
    ccatmo_c12_mode: mode of the total number of CCatmo background events on C12 (np.array of float)
    ccatmo_c12_50: mean of ccatmo_c12_mode (float)
    ccatmo_c12_50_sigma: std of ccatmo_c12_mode (float)
    ccatmo_c12_50_16: 16% C.L. of ccatmo_c12_mode (float)
    ccatmo_c12_50_84: 84% C.L. of ccatmo_c12_mode (float)
    ccatmo_c12_expected: number of expected CCatmo background events on C12 in the energy window (float)

    """

    """ Variable, which defines the date and time of running the script: """
    # get the date and time, when the script was run:
    date = datetime.datetime.now()
    now = date.strftime("%Y-%m-%d %H:%M")

    # calculate the number of files, that are read in (Equivalent to the number of entries in the result array)
    # (integer):
    number_of_entries = file_number_stop - file_number_start + 1

    """ Preallocate the arrays, where the results of the analysis of the different datasets should be appended: """
    # mode of the total number of signal events (np.array of float):
    s_mode = np.array([])
    # values of upper 90 percent limit of the number of signal events (np.array of float):
    s_90_limit = np.array([])
    # mode of the total number of DSNB background events (np.array of float):
    dsnb_mode = np.array([])
    # mode of the total number of CCatmo background events on protons (np.array of float):
    ccatmo_p_mode = np.array([])
    # mode of the total number of reactor background events (np.array of float):
    reactor_mode = np.array([])
    # mode of the total number of NCatmo background events (np.array of float):
    ncatmo_mode = np.array([])
    # mode of the total number of CCatmo background events on C12 (np.array of float):
    ccatmo_c12_mode = np.array([])

    """ Read in the files, where the results of the analysis are saved and read the result-values: """
    for number in np.arange(file_number_start, file_number_stop + 1, 1):
        # load the file corresponding to Dataset{number}_mcmc_analysis.txt:
        result_analysis = np.loadtxt(path_analysis + "/Dataset{0:d}_mcmc_analysis.txt".format(number))
        # get value of mode of S (float):
        value_mode_s = result_analysis[0]
        # get value of 90 percent limit S (float):
        value_s_90_limit = result_analysis[1]
        # get value of mode of B_DSNB (float):
        value_mode_dsnb = result_analysis[2]
        # get value of mode of B_CCatmo_p (float):
        value_mode_ccatmo_p = result_analysis[3]
        # get value of mode of B_CCatmo_c12 (float):
        value_mode_ccatmo_c12 = result_analysis[4]
        # get value of mode of B_reactor (float):
        value_mode_reactor = result_analysis[5]
        # get value of mode of B_NCatmo (float):
        value_mode_ncatmo = result_analysis[6]

        # Append the values to the arrays (np.array of float):
        s_mode = np.append(s_mode, value_mode_s)
        s_90_limit = np.append(s_90_limit, value_s_90_limit)
        dsnb_mode = np.append(dsnb_mode, value_mode_dsnb)
        ccatmo_p_mode = np.append(ccatmo_p_mode, value_mode_ccatmo_p)
        reactor_mode = np.append(reactor_mode, value_mode_reactor)
        ncatmo_mode = np.append(ncatmo_mode, value_mode_ncatmo)
        ccatmo_c12_mode = np.append(ccatmo_c12_mode, value_mode_ccatmo_c12)

    """ Calculate the mean and probability interval: """
    # calculate the mean and 16% and 84% confidence level and 2.5% and 97.5% CL and 0.15% and 99.85% CL of the
    # array S_mode (float):
    s_50 = np.mean(s_mode)
    s_50_sigma = np.std(s_mode)
    s_50_16, s_50_84 = np.percentile(s_mode, [16, 84])
    s_50_2_5, s_50_97_5 = np.percentile(s_mode, [2.5, 97.5])
    s_50_0_15, s_50_99_85 = np.percentile(s_mode, [0.15, 99.85])

    # calculate the mean and 16% and 84% confidence level of the array s_90_limit (float):
    s_90 = np.mean(s_90_limit)
    s_90_sigma = np.std(s_90_limit)
    s_90_16, s_90_84 = np.percentile(s_90_limit, [16, 84])
    s_90_2_5, s_90_97_5 = np.percentile(s_90_limit, [2.5, 97.5])
    s_90_0_15, s_90_99_85 = np.percentile(s_90_limit, [0.15, 99.85])

    # calculate the mean and 16% and 84% confidence level of the array DSNB_mode (float):
    dsnb_50 = np.mean(dsnb_mode)
    dsnb_50_sigma = np.std(dsnb_mode)
    dsnb_50_16, dsnb_50_84 = np.percentile(dsnb_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array CCatmo_p_mode (float):
    ccatmo_p_50 = np.mean(ccatmo_p_mode)
    ccatmo_p_50_sigma = np.std(ccatmo_p_mode)
    ccatmo_p_50_16, ccatmo_p_50_84 = np.percentile(ccatmo_p_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array Reactor_mode (float):
    reactor_50 = np.mean(reactor_mode)
    reactor_50_sigma = np.std(reactor_mode)
    reactor_50_16, reactor_50_84 = np.percentile(reactor_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array NCatmo_mode (float):
    ncatmo_50 = np.mean(ncatmo_mode)
    ncatmo_50_sigma = np.std(ncatmo_mode)
    ncatmo_50_16, ncatmo_50_84 = np.percentile(ncatmo_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array CCatmo_c12_mode (float):
    ccatmo_c12_50 = np.mean(ccatmo_c12_mode)
    ccatmo_c12_50_sigma = np.std(ccatmo_c12_mode)
    ccatmo_c12_50_16, ccatmo_c12_50_84 = np.percentile(ccatmo_c12_mode, [16, 84])

    """ Load the analysis information file to get the expected/true number of events in the energy window and to get the 
        energy window: """
    # load the txt file (np.array of float):
    information_analysis = np.loadtxt(file_info_analysis)
    # lower bound of the energy window in MeV (float):
    lower_energy_bound = information_analysis[1]
    # upper bound of the energy window in MeV (float):
    upper_energy_bound = information_analysis[2]
    # number of expected signal events in the energy window (float):
    signal_expected = information_analysis[4]
    # number of expected DSNB backgrounds events in the energy window (float):
    dsnb_expected = information_analysis[6]
    # number of expected atmospheric CC background events on protons in the energy window (float):
    ccatmo_p_expected = information_analysis[7]
    # number of expected reactor background events in the energy window (float):
    reactor_expected = information_analysis[9]
    # number of expected atmospheric NC background events in the energy window (float):
    ncatmo_expected = information_analysis[10]
    # number of expected atmopsheric CC background events on C12 in the energy window (float):
    ccatmo_c12_expected = information_analysis[10]

    if save_data:
        np.savetxt(path_output + "/result_mcmc/result_dataset_output_{0:d}.txt".format(dm_mass),
                   np.array([lower_energy_bound, upper_energy_bound, number_of_entries,
                             signal_expected, s_50, s_50_sigma, s_50_16, s_50_84,
                             s_90, s_90_sigma, s_90_16, s_90_84,
                             dsnb_expected, dsnb_50, dsnb_50_sigma, dsnb_50_16, dsnb_50_84,
                             ccatmo_p_expected, ccatmo_p_50, ccatmo_p_50_sigma, ccatmo_p_50_16, ccatmo_p_50_84,
                             reactor_expected, reactor_50, reactor_50_sigma, reactor_50_16, reactor_50_84,
                             ncatmo_expected, ncatmo_50, ncatmo_50_sigma, ncatmo_50_16, ncatmo_50_84,
                             ccatmo_c12_expected, ccatmo_c12_50, ccatmo_c12_50_sigma, ccatmo_c12_50_16,
                             ccatmo_c12_50_84,
                             s_50_2_5, s_50_97_5, s_50_0_15, s_50_99_85,
                             s_90_2_5, s_90_97_5, s_90_0_15, s_90_99_85]),
                   fmt="%4.5f",
                   header="Results of the analysis of the spectra in dataset_output_{3} "
                          "(with output_analysis_v7_server.py, {0}):\n"
                          "Analysis of Dataset_{1:d}.txt to Dataset_{2:d}.txt\n"
                          "\n"
                          "Information to the values below:\n"
                          "Lower bound of the energy window in MeV, upper bound of the energy window in MeV\n"
                          "Number of datasets that were analyzed,\n"
                          "Expected number of signal events from simulation,\n"
                          "Mean of the observed number of signal events,\n"
                          "Standard deviation of the observed number of signal events,\n"
                          "16 % confidence level of the observed number of signal events,\n"
                          "84 % confidence level of the observed number of signal events,\n"
                          "Mean of the 90% probability limit of the observed number of signal events,\n"
                          "Standard deviation of the 90% probability limit of the observed number of signal events,\n"
                          "16 % confidence level of the 90% probability limit of the observed number of signal "
                          "events,\n"
                          "84 % confidence level of the 90% probability limit of the observed number of signal "
                          "events,\n"
                          "Expected number of DSNB background events from simulation,\n"
                          "Mean of the observed number of DSNB background events,\n"
                          "Standard deviation of the observed number of DSNB background events,\n"
                          "16 % confidence level of the observed number of DSNB background events,\n"
                          "84 % confidence level of the observed number of DSNB background events,\n"
                          "Expected number of CCatmo background events on p from simulation,\n"
                          "Mean of the observed number of atmo. CC background events on p,\n"
                          "Standard deviation of the observed number of atmo. CC background events on p,\n"
                          "16 % confidence level of the observed number of atmo. CC background events on p,\n"
                          "84 % confidence level of the observed number of atmo. CC background events on p,\n"
                          "Expected number of reactor background events from simulation,\n"
                          "Mean of the observed number of Reactor background events,\n"
                          "Standard deviation of the observed number of Reactor background events,\n"
                          "16 % confidence level of the observed number of Reactor background events,\n"
                          "84 % confidence level of the observed number of Reactor background events,\n"
                          "Expected number of NCatmo background events from simulation,\n"
                          "Mean of the observed number of atmo. NC background events,\n"
                          "Standard deviation of the observed number of atmo. NC background events,\n"
                          "16 % confidence level of the observed number of atmo. NC background events,\n"
                          "84 % confidence level of the observed number of atmo. NC background events,\n"
                          "Expected number of CCatmo background events on C12 from simulation,\n"
                          "Mean of the observed number of atmo. CC background events on C12,\n"
                          "Standard deviation of the observed number of atmo. CC background events on C12,\n"
                          "16 % confidence level of the observed number of atmo. CC background events on C12,\n"
                          "84 % confidence level of the observed number of atmo. CC background events on C12,\n"
                          "2.5 % confidence level of the observed number of signal events,\n"
                          "97.5 % confidence level of the observed number of signal events,\n"
                          "0.15 % confidence level of the observed number of signal events,\n"
                          "99.85 % confidence level of the observed number of signal events,\n"
                          "2.5 % confidence level of the 90% probability limit of the observed number of signal "
                          "events,\n"
                          "97.5 % confidence level of the 90% probability limit of the observed number of signal "
                          "events,\n"
                          "0.15 % confidence level of the 90% probability limit of the observed number of signal "
                          "events,\n"
                          "99.85 % confidence level of the 90% probability limit of the observed number of signal "
                          "events,\n:"
                   .format(now, file_number_start, file_number_stop, dm_mass))

        # print message, that result data is saved in file:
        print("result data is saved in the file result_dataset_output_{0:d}.txt".format(dm_mass))

    return (number_of_entries, lower_energy_bound, upper_energy_bound,
            s_mode, s_50, s_50_sigma, s_50_16, s_50_84, signal_expected,
            s_90_limit, s_90, s_90_sigma, s_90_16, s_90_84,
            dsnb_mode, dsnb_50, dsnb_50_sigma, dsnb_50_16, dsnb_50_84, dsnb_expected,
            ccatmo_p_mode, ccatmo_p_50, ccatmo_p_50_sigma, ccatmo_p_50_16, ccatmo_p_50_84, ccatmo_p_expected,
            reactor_mode, reactor_50, reactor_50_sigma, reactor_50_16, reactor_50_84, reactor_expected,
            ncatmo_mode, ncatmo_50, ncatmo_50_sigma, ncatmo_50_16, ncatmo_50_84, ncatmo_expected,
            ccatmo_c12_mode, ccatmo_c12_50, ccatmo_c12_50_sigma, ccatmo_c12_50_16, ccatmo_c12_50_84,
            ccatmo_c12_expected,
            s_50_2_5, s_50_97_5, s_50_0_15, s_50_99_85, s_90_2_5, s_90_97_5, s_90_0_15, s_90_99_85)
