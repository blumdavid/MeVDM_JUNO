#!/junofs/users/dblum/my_env/bin/python
# author: David Blum

""" auto_gen_dataset.sh is a script to automate the dataset generation of the indirect MeV-Dark Matter search
    with JUNO!

    INPUT: the simulated spectra of the signal for different DM masses and the different backgrounds

    Script: the script generates datasets corresponding to the simulated spectra FOR SEVERAL MASSES

    The script should be used on the IHEP cluster!
"""

# import of the necessary packages:
import os
import numpy as np
import subprocess


# define the Dark matter masses in MeV, who's datasets should be generated and that should be analyzed
# (np.array of float):
# TODO: set the DM masses that should be analyzed ("masses"):
masses = np.array([25, 35, 45, 55, 65, 75, 85, 95])

# loop over the DM masses:
for DM_mass in masses:

    # define the names od the different folders
    # TODO: check the directory "path_folder"
    path_folder = "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"
    path_dataset_output = "dataset_output_{0}".format(DM_mass)
    path_dataset = "datasets"
    path_result = "result_mcmc"
    path_analysis = "analysis_mcmc"

    # first create the necessary directories:
    os.mkdir(path_dataset_output)
    os.mkdir(path_dataset_output + "/" + path_dataset)
    os.mkdir(path_dataset_output + "/" + path_result)
    os.mkdir(path_dataset_output + "/" + path_analysis)

    # Number of dataset, that should be generated (from dataset_start to dataset_stop) (integer):
    # TODO: set "dataset_start" and "dataset_stop":
    dataset_start = 1
    dataset_stop = 10000

    # directory of the simulated spectra:
    # TODO: set the directory and folder names of the simulated spectra:
    path_simu = "/junofs/users/dblum/work/simu_spectra"
    # file names of the simulated spectra used:
    filename_DSNB = "DSNB_EmeanNuXbar22_bin100keV.txt"
    filename_DSNB_info = "DSNB_info_EmeanNuXbar22_bin100keV.txt"
    filename_CCatmo = "CCatmo_Osc1_bin100keV.txt"
    filename_CCatmo_info = "CCatmo_info_Osc1_bin100keV.txt"
    filename_reactor = "Reactor_NH_power36_bin100keV.txt"
    filename_reactor_info = "Reactor_info_NH_power36_bin100keV.txt"

    # execute the python script gen_dataset_v2_server.py with the correct input arguments to generate datasets
    # from the simulated spectra:
    """ Description of the arguments for the gen_dataset_v1_server.py script:
        - sys.argv[0] path to the script = gen_dataset_v1_server.py
        - sys.argv[1] Dark matter mass in MeV
        - sys.argv[2] dataset_start
        - sys.argv[3] dataset_stop
        - sys.argv[4] directory of the correct folder = /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor
        - sys.argv[5] dataset_output folder = dataset_output_{DM_mass}
        - sys.argv[6] datasets folder = datasets
        - sys.argv[7] directory of the simulated spectra = /junofs/users/dblum/work/simu_spectra
        - sys.argv[8] file name of DSNB spectrum = DSNB_EmeanNuXbar22_bin100keV.txt
        - sys.argv[9] file name of DSNB info = DSNB_info_EmeanNuXbar22_bin100keV.txt
        - sys.argv[10] file name of CCatmo spectrum = CCatmo_Osc1_bin100keV.txt
        - sys.argv[11] file name of CCatmo info = CCatmo_info_Osc1_bin100keV.txt
        - sys.argv[12] file name of reactor spectrum = Reactor_NH_power36_bin100keV.txt
        - sys.argv[13] file name of reactor info = Reactor_info_NH_power36_bin100keV.txt
    """

    # Run gen_dataset_v1_server.py with the correct arguments:
    gen_dataset_process = subprocess.Popen(['python gen_dataset_v1_server.py %d %d %d %s %s %s %s %s %s %s %s %s %s'
                                            %(DM_mass, dataset_start, dataset_stop, path_folder, path_dataset_output,
                                              path_dataset, path_simu, filename_DSNB, filename_DSNB_info,
                                              filename_CCatmo, filename_CCatmo_info, filename_reactor,
                                              filename_reactor_info)], shell=True)
    # wait until the process is finished:
    gen_dataset_status = gen_dataset_process.wait()

    # print message, that the datasets for one DM mass are generated:
    print("datasets for DM mass = {0} MeV are generated".format(DM_mass))

print("Datasets are generated!")







