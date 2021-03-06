#!/junofs/users/dblum/miniconda3/envs/miniconda_env/bin/python
# author: David Blum

""" Script "auto_output_analysis_v3.py" from 12.05.2020.
    It was used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo".

    auto_output_analysis_v3.py is a script to automate the display of the analysis of the indirect MeV-Dark Matter search
    with JUNO!

    INPUT: the simulated spectra of the signal for different DM masses and the different backgrounds

    Script: the script displays the results of the MCMC analysis FOR SEVERAL MASSES

    The script should be used on the IHEP cluster!
"""

# import of the necessary packages:
import numpy as np
import subprocess

# TODO: Check if "output_analysis_v5.py", "output_analysis_v5_server.py" and "auto_output_analysis_v3.py" in same folder

# define the Dark matter masses in MeV, which should be displayed (np.array of float):
# TODO: set the DM masses that should be analyzed ("masses"):
masses = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

# loop over the DM masses:
for DM_mass in masses:

    # define the names of the different directories and folders:
    # TODO: check the directory "path_folder"
    path_folder = "/junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo"
    # path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN/simulation_1"
    path_dataset_output = "dataset_output_{0}".format(DM_mass)

    # Number of files (equal to number of datasets per DM mass), that should be displayed
    # (from dataset_start to dataset_stop) (integer):
    # TODO: set "dataset_start" and "dataset_stop":
    dataset_start = 0
    # dataset_stop = 9999
    dataset_stop = 999

    # Number of datasets analyzed per job:
    # TODO: set the number of datasets analyzed per job!
    # number_of_datasets_per_job = 100
    number_of_datasets_per_job = 10

    # execute the python script output_analysis_v5_server.py with the correct input arguments to generate datasets
    # from the simulated spectra:
    """ Description of the arguments for the output_analysis_v5_server.py script:
        - sys.argv[0] name of the script = output_analysis_v5_server.py
        - sys.argv[1] Dark matter mass in MeV
        - sys.argv[2] directory of the correct folder = "/junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo"
        - sys.argv[3] dataset_output folder = "dataset_output_{DM_mass}"
        - sys.argv[4] number of datasets analyzed per job = number_of_datasets_per_job
        - sys.argv[5] dataset_start
        - sys.argv[6] dataset_stop
    """

    # Run output_analysis_v5_server.py with the correct arguments:
    output_analysis_process = subprocess.Popen(['python output_analysis_v5_server.py %d %s %s %d %d %d'
                                                %(DM_mass, path_folder, path_dataset_output, number_of_datasets_per_job,
                                                  dataset_start, dataset_stop)], shell=True)

    # wait until the process is finished:
    output_analysis_status = output_analysis_process.wait()

print("The output of the analysis is saved!")



