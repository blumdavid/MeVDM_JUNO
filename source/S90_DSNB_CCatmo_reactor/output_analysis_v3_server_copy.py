""" This script is a copy of the script "output_analysis_v3_server.py" from 28.09.2018.
    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor".


    Script to display and analyze the results of the MCMC analysis on the IHEP cluster in China
    done with analyze_spectra_v4_server2.py!

    Script uses the function output_analysis() from output_analysis_v3.py,
    but is optimized to run on the local computer.

    In analyze_spectra_v4_server2.py the dataset and the simulated spectrum were analyzed with
    Markov Chain Monte Carlo (MCMC) sampling and the results of the analysis are saved in the
    files DatasetX_mcmc_analysis.txt

    In analyze_spectra_v4_server2.py the mode of the total number of signal and background events are determined by
    MCMC sampling of the posterior probability

    give 7 arguments to the script:
    - sys.argv[0] name of the script = output_analysis_v3_server.py
    - sys.argv[1] Dark matter mass in MeV
    - sys.argv[2] directory of the correct folder = "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"
    - sys.argv[3] dataset_output folder = "dataset_output_{DM_mass}"
    - sys.argv[4] number of datasets analyzed per job
    - sys.argv[5] dataset_start
    - sys.argv[6] dataset_stop
"""

import numpy as np
import sys
from output_analysis_v3 import output_analysis

""" Set boolean value to define, if the result of output_analysis_v3.py is saved: """
SAVE_DATA = True

""" get the DM mass in MeV (float): """
DM_mass = int(sys.argv[1])

""" set the path to the correct folder: """
path_folder = str(sys.argv[2])

""" set the path of the output folder: """
path_output = path_folder + "/" + str(sys.argv[3])

""" set the path of the analysis folder: """
path_analysis = path_output + "/analysis_mcmc"

""" set the number of datasets analyzed per job: """
number_of_datasets_per_job = int(sys.argv[4])

""" Set the path of the file, where the information about the analysis is saved: """
file_info_analysis = path_analysis + "/info_mcmc_analysis_1_{0}.txt".format(number_of_datasets_per_job)

# Set the number of the files, that should be read in (is equal to dataset_start and dataset_stop):
file_number_start = int(sys.argv[5])
file_number_stop = int(sys.argv[6])


""" display and analyze the results from the analysis with function output_analysis() from output_analysis_v3.py """
(number_of_entries, lower_energy_bound, upper_energy_bound, S_mode, S_50, S_50_sigma, S_50_16, S_50_84, signal_expected,
 S_90_limit, S_90, S_90_sigma, S_90_16, S_90_84, DSNB_mode, DSNB_50, DSNB_50_sigma, DSNB_50_16, DSNB_50_84,
 DSNB_expected, CCatmo_mode, CCatmo_50, CCatmo_50_sigma, CCatmo_50_16, CCatmo_50_84, CCatmo_expected, Reactor_mode,
 Reactor_50, Reactor_50_sigma, Reactor_50_16, Reactor_50_84, Reactor_expected, S_50_2_5, S_50_97_5, S_50_0_15,
 S_50_99_85, S_90_2_5, S_90_97_5, S_90_0_15, S_90_99_85) \
    = output_analysis(SAVE_DATA, DM_mass, path_output, path_analysis, file_info_analysis, file_number_start,
                      file_number_stop)


if SAVE_DATA:
    # Save the array S_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/S_mode_DMmass{0}.txt".format(DM_mass),
               S_mode, fmt="%4.5f",
               header="Observed number of signal events (mode of S) from DM with mass={2:.1f}MeV in virtual "
                      "experiments\n (in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound, DM_mass))

    # Save the array S_90_limit to txt file:
    np.savetxt(path_output + "/result_mcmc/S_90_limit_DMmass{0}.txt".format(DM_mass),
               S_90_limit, fmt="%4.5f",
               header="90 % upper limit of number of observed signal events S from DM with mass={0:.1f}MeV in virtual"
                      "experiments"
               .format(DM_mass))

    # Save the array DSNB_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/DSNB_mode_DMmass{0}.txt".format(DM_mass),
               DSNB_mode, fmt="%4.5f",
               header="Observed number of DSNB background events (mode of B_DSNB) in virtual experiments\n"
                      "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound))

    # Save the array CCatmo_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/CCatmo_mode_DMmass{0}.txt".format(DM_mass),
               CCatmo_mode, fmt="%4.5f",
               header="Observed number of CCatmo background events (mode of B_CCatmo) in virtual experiments\n"
                      "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound))

    # Save the array Reactor_mode to txt file:
    np.savetxt(path_output + "/result_mcmc/Reactor_mode_DMmass{0}.txt".format(DM_mass),
               Reactor_mode, fmt="%4.5f",
               header="Observed number of reactor background events (mode of B_reactor) in virtual experiments\n"
                      "(in the energy window from {0:.1f} MeV to {1:.1f} MeV)"
               .format(lower_energy_bound, upper_energy_bound))
