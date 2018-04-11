#!/junofs/users/dblum/my_env/bin/python
# author: David Blum

""" Script to check the status of the submitted jobs at the IHEP cluster:

"""
import numpy as np
import os
import time

# define the Dark matter masses in MeV (np.array of float):
# TODO: set the DM masses that should be analyzed ("masses"):
DM_mass = np.array([25, 35, 45, 55, 65, 75, 85, 95])

# directory to the simulation (string):
# TODO: check the directory "path_folder"
path_folder = "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"

# folder, where the analysis files are saved (string):
path_analysis = "/analysis_mcmc"

# Set the number of datasets, which were generated per DM mass (integer):
# TODO: check if Number_dataset is correct!
Number_dataset = 10000

# ONE info_mcmc_analysis_{}_{}.txt file and one acceptance_fraction_{}_{}.txt file is generated per Job during the
# analysis (integer):
Number_output = 2

# Number of jobs, that were submitted per DM mass (integer):
# TODO: set the number of jobs!
Number_jobs_per_mass = 100

# Number of files, that should exist in the directory "/analysis_mcmc", when the jobs are finished (integer):
Stop_criterion = Number_dataset + Number_jobs_per_mass * Number_output


def check_file_number(directory):
    """
    function, which calculates the number of files in a given directory

    :param directory: directory or path, where the files are saved (string)
    :return: num_actual: actual number of files in the directory (integer)
    """
    # count the number of file in the directory given by dir (integer):
    num_actual = len(os.walk(directory).__next__()[2])

    return num_actual


# go through the different masses:
for mass in DM_mass:
    # define the directory, where the analysis is saved, for the given DM mass (string):
    path = path_folder + "/dataset_output_{0}".format(mass) + path_analysis

    # start point of the while loop (integer):
    Number_actual = 0

    while Number_actual < Stop_criterion:

        # calculate the actual number of files in the path (integer=:
        Number_actual = check_file_number(path)

        if Number_actual < Stop_criterion:
            print("jobs for DM mass = {0} are not finished...".format(mass))
            # if correct number is not reached -> wait a certain time and check again:
            time.sleep(60*10)
        else:
            # if correct number is reached -> print info and go to the folder of the next mass:
            print("Analysis of DM mass = {0} MeV is finished".format(mass))
