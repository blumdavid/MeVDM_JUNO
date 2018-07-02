""" output_analysis_v3.py:

    The Script is a function to display and analyze the results of the MCMC analysis.

    The MCMC analysis is done either with analyze_spectra_v4_local.py or analyze_spectra_v4_server.py,
    OR with analyze_spectra_v4_server2.py.

    The script output_analysis_v3.py is used in the scripts output_analysis_v3_local.py
    (when you display the results on the local computer) and in output_analysis_v3_server.py
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
    signal_expected: number of expected signal events in the energy window (float)
    s_90_limit: values of upper 90 percent limit of the number of signal events (np.array of float)
    s_90: mean of s_90_limit (float)
    s_90_sigma: std of s_90_limit (float)
    s_90_16: 16% confidence level of s_90_limit (float)
    s_90_84: 84% confidence level of s_90_limit (float)
    dsnb_mode: mode of the total number of DSNB background events (np.array of float)
    dsnb_50: mean of dsnb_mode (float)
    dsnb_50_sigma: std of dsnb_mode (float)
    dsnb_50_16: 16% confidence level of dsnb_mode (float)
    dsnb_50_84: 84% confidence level of dsnb_mode (float)
    dsnb_expected: number of expected DSNB background events in the energy window (float)
    ccatmo_mode: mode of the total number of CCatmo background events (np.array of float)
    ccatmo_50: mean of ccatmo_mode (float)
    ccatmo_50_sigma: std of ccatmo_mode (float)
    ccatmo_50_16: 16% C.L. of ccatmo_mode (float)
    ccatmo_50_84: 84% C.L. of ccatmo_mode (float)
    ccatmo_expected: number of expected atmospheric CC background events in the energy window (float)
    reactor_mode: mode of the total number of reactor background events (np.array of float)
    reactor_50: mean of reactor_mode (float)
    reactor_50_sigma: std of reactor_mode (float)
    reactor_50_16: 16% C.L. of reactor_mode (float)
    reactor_50_84: 84% C.L. of reactor_mode (float)
    reactor_expected: number of expected reactor background events in the energy window (float)

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
    # mode of the total number of CCatmo background events (np.array of float):
    ccatmo_mode = np.array([])
    # mode of the total number of reactor background events (np.array of float):
    reactor_mode = np.array([])

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
        # get value of mode of B_CCatmo (float):
        value_mode_ccatmo = result_analysis[3]
        # get value of mode of B_reactor (float):
        value_mode_reactor = result_analysis[4]

        # Append the values to the arrays (np.array of float):
        s_mode = np.append(s_mode, value_mode_s)
        s_90_limit = np.append(s_90_limit, value_s_90_limit)
        dsnb_mode = np.append(dsnb_mode, value_mode_dsnb)
        ccatmo_mode = np.append(ccatmo_mode, value_mode_ccatmo)
        reactor_mode = np.append(reactor_mode, value_mode_reactor)

    """ Calculate the mean and probability interval: """
    # calculate the mean and 16% and 84% confidence level of the array S_mode (float):
    s_50 = np.mean(s_mode)
    s_50_sigma = np.std(s_mode)
    s_50_16, s_50_84 = np.percentile(s_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array s_90_limit (float):
    s_90 = np.mean(s_90_limit)
    s_90_sigma = np.std(s_90_limit)
    s_90_16, s_90_84 = np.percentile(s_90_limit, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array DSNB_mean (float):
    dsnb_50 = np.mean(dsnb_mode)
    dsnb_50_sigma = np.std(dsnb_mode)
    dsnb_50_16, dsnb_50_84 = np.percentile(dsnb_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array CCatmo_mean (float):
    ccatmo_50 = np.mean(ccatmo_mode)
    ccatmo_50_sigma = np.std(ccatmo_mode)
    ccatmo_50_16, ccatmo_50_84 = np.percentile(ccatmo_mode, [16, 84])

    # calculate the mean and 16% and 84% confidence level of the array Reactor_mean (float):
    reactor_50 = np.mean(reactor_mode)
    reactor_50_sigma = np.std(reactor_mode)
    reactor_50_16, reactor_50_84 = np.percentile(reactor_mode, [16, 84])

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
    # number of expected atmospheric CC background events in the energy window (float):
    ccatmo_expected = information_analysis[7]
    # number of expected reactor background events in the energy window (float):
    reactor_expected = information_analysis[8]

    if save_data:
        np.savetxt(path_output + "/result_mcmc/result_dataset_output_{0:d}.txt".format(dm_mass),
                   np.array([lower_energy_bound, upper_energy_bound, number_of_entries,
                             signal_expected, s_50, s_50_sigma, s_50_16, s_50_84,
                             s_90, s_90_sigma, s_90_16, s_90_84,
                             dsnb_expected, dsnb_50, dsnb_50_sigma, dsnb_50_16, dsnb_50_84,
                             ccatmo_expected, ccatmo_50, ccatmo_50_sigma, ccatmo_50_16, ccatmo_50_84,
                             reactor_expected, reactor_50, reactor_50_sigma, reactor_50_16, reactor_50_84]),
                   fmt="%4.5f",
                   header="Results of the analysis of the spectra in dataset_output_{3} "
                          "(with output_analysis_v3.py, {0}):\n"
                          "Analysis of Dataset_{1:d}.txt to Dataset_{2:d}.txt\n"
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
                          "Expected number of CCatmo background events from simulation,\n"
                          "Mean of the observed number of atmo. CC background events,\n"
                          "Standard deviation of the observed number of atmo. CC background events,\n"
                          "16 % confidence level of the observed number of atmo. CC background events,\n"
                          "84 % confidence level of the observed number of atmo. CC background events,\n"
                          "Expected number of reactor background events from simulation,\n"
                          "Mean of the observed number of Reactor background events,\n"
                          "Standard deviation of the observed number of Reactor background events,\n"
                          "16 % confidence level of the observed number of Reactor background events,\n"
                          "84 % confidence level of the observed number of Reactor background events:"
                   .format(now, file_number_start, file_number_stop, dm_mass))

        # print message, that result data is saved in file:
        print("result data is saved in the file result_dataset_output_{0:d}.txt".format(dm_mass))

    return (number_of_entries, lower_energy_bound, upper_energy_bound,
            s_mode, s_50, s_50_sigma, s_50_16, s_50_84, signal_expected,
            s_90_limit, s_90, s_90_sigma, s_90_16, s_90_84,
            dsnb_mode, dsnb_50, dsnb_50_sigma, dsnb_50_16, dsnb_50_84, dsnb_expected,
            ccatmo_mode, ccatmo_50, ccatmo_50_sigma, ccatmo_50_16, ccatmo_50_84, ccatmo_expected,
            reactor_mode, reactor_50, reactor_50_sigma, reactor_50_16, reactor_50_84, reactor_expected)
