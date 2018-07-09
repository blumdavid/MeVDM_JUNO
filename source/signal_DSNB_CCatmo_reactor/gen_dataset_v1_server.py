""" Script to generate several data-sets (virtual experiments) from the spectra,
    which are calculated with gen_spectrum_v2.py (VERSION 1) on the IHEP cluster:

    Script uses the function gen_dataset() from gen_dataset_v1.py,
    but is optimized to run on the IHEP cluster.

    give 14 arguments to the script:
    - sys.argv[0] name of the script = gen_dataset_v1_server.py
    - sys.argv[1] Dark matter mass in MeV
    - sys.argv[2] dataset_start
    - sys.argv[3] dataset_stop
    - sys.argv[4] directory of the correct folder = "/junofs/users/dblum/work/S90_DSNB_CCatmo_reactor"
    - sys.argv[5] dataset_output folder = "dataset_output_{DM_mass}"
    - sys.argv[6] datasets folder = "datasets"
    - sys.argv[7] directory of the simulated spectra = "/junofs/users/dblum/work/simu_spectra"
    - sys.argv[8] file name of DSNB spectrum = "DSNB_EmeanNuXbar22_bin100keV.txt"
    - sys.argv[9] file name of DSNB info = "DSNB_info_EmeanNuXbar22_bin100keV.txt"
    - sys.argv[10] file name of CCatmo spectrum = "CCatmo_Osc1_bin100keV.txt"
    - sys.argv[11] file name of CCatmo info = "CCatmo_info_Osc1_bin100keV.txt"
    - sys.argv[12] file name of reactor spectrum = "Reactor_NH_power36_bin100keV.txt"
    - sys.argv[13] file name of reactor info = "Reactor_info_NH_power36_bin100keV.txt"
"""

# import of the necessary packages:
import sys
from gen_dataset_v1 import gen_dataset

""" get the DM mass in MeV (float): """
DM_mass = int(sys.argv[1])

""" Define parameters for saving of the virtual experiments: 
    number_dataset (dataset_stop - dataset_start) defines, how many datasets (virtual experiments)
    are generated. """
# INFO-me: 10000 datasets might be a good number for the analysis
# dataset_start defines the start point (integer):
dataset_start = int(sys.argv[2])
# dataset_stop defines the end point (integer):
dataset_stop = int(sys.argv[3])

""" Set boolean variable, which controls, if the txt-file with the data are saved or not (should be True): """
SAVE_DATA_TXT = True

""" Set boolean variable, which controls, if the all files (txt and png) with the data are saved or not 
    (should be False): """
SAVE_DATA_ALL = False

""" Set boolean variable, which controls, if the data is displayed on screen or not (should be False): """
DISPLAY_DATA = False

""" set the path to the correct folder: """
path_folder = str(sys.argv[4])

""" set the path of the output folder: """
path_output = path_folder + "/" + str(sys.argv[5])

""" set the path of the folder, where the datasets are saved: """
path_dataset = path_output + "/" + str(sys.argv[6])

""" Files, that are read in to generate data-sets (virtual experiments): """
path_simu = str(sys.argv[7])
# TODO: check the file of the simulated signal spectrum:
file_signal = path_simu + "/{0}signal_bin100keV.txt".format(DM_mass)
file_signal_info = path_simu + "/{0}signal_info_bin100keV.txt".format(DM_mass)
file_DSNB = path_simu + "/" + str(sys.argv[8])
file_DSNB_info = path_simu + "/" + str(sys.argv[9])
file_CCatmo = path_simu + "/" + str(sys.argv[10])
file_CCatmo_info = path_simu + "/" + str(sys.argv[11])
file_reactor = path_simu + "/" + str(sys.argv[12])
file_reactor_info = path_simu + "/" + str(sys.argv[13])

""" generate the datasets with function gen_dataset() from gen_dataset_v1.py: """
gen_dataset(DM_mass, SAVE_DATA_TXT, SAVE_DATA_ALL, DISPLAY_DATA, dataset_start, dataset_stop, path_output,
            path_dataset, file_signal, file_signal_info, file_DSNB, file_DSNB_info, file_CCatmo, file_CCatmo_info,
            file_reactor, file_reactor_info)

