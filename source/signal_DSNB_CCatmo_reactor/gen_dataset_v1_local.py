""" Script to generate several data-sets (virtual experiments) from the spectra,
    which are calculated with gen_spectrum_v2.py (VERSION 1):

    Script uses the function gen_dataset() from gen_dataset_v1.py,
    but is optimized to run on the local computer.

"""

# import of the necessary packages:
from work.MeVDM_JUNO.source.signal_DSNB_CCatmo_reactor.gen_dataset_v1 import gen_dataset

""" Set the DM mass in MeV (float): """
DM_mass = 20

""" Define parameters for saving of the virtual experiments: 
    number_dataset (dataset_stop - dataset_start) defines, how many datasets (virtual experiments)
    are generated. """
# INFO-me: 10000 datasets might be a good number for the analysis
# dataset_start defines the start point (integer):
dataset_start = 1
# dataset_stop defines the end point (integer):
dataset_stop = 1

""" Set boolean variable, which controls, if the txt-file with the data are saved or not: """
SAVE_DATA_TXT = True

""" Set boolean variable, which controls, if the all files (txt and png) with the data are saved or not: """
SAVE_DATA_ALL = True

""" Set boolean variable, which controls, if the data is displayed on screen or not (should be False): """
# display the simulated spectra (when True, you get a warning (can't invoke "event" command: application has been
# destroyed while executing "event generate $w <<ThemeChanged>>" (procedure "ttk::ThemeChanged" line 6) invoked from
# within "ttk::ThemeChanged") -> there is a conflict between the pyplot.close() command and the pyplot.show() command.
# BUT: datasets are generated correctly!!!
DISPLAY_DATA = False

""" set the path to the correct folder: """
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/signal_DSNB_CCatmo_reactor"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_{0}".format(DM_mass)

""" set the path of the folder, where the datasets are saved: """
path_dataset = path_output + "/datasets"

""" Files, that are read in to generate data-sets (virtual experiments): """
path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2"
file_signal = path_simu + "/signal_DMmass{0}_bin100keV.txt".format(DM_mass)
file_signal_info = path_simu + "/signal_info_DMmass{0}_bin100keV.txt".format(DM_mass)
file_DSNB = path_simu + "/DSNB_EmeanNuXbar22_bin100keV.txt"
file_DSNB_info = path_simu + "/DSNB_info_EmeanNuXbar22_bin100keV.txt"
file_CCatmo = path_simu + "/CCatmo_Osc1_bin100keV.txt"
file_CCatmo_info = path_simu + "/CCatmo_info_Osc1_bin100keV.txt"
file_reactor = path_simu + "/Reactor_NH_power36_bin100keV.txt"
file_reactor_info = path_simu + "/Reactor_info_NH_power36_bin100keV.txt"

""" generate the datasets with function gen_dataset() from gen_dataset_v1.py: """
gen_dataset(DM_mass, SAVE_DATA_TXT, SAVE_DATA_ALL, DISPLAY_DATA, dataset_start, dataset_stop, path_output,
            path_dataset, file_signal, file_signal_info, file_DSNB, file_DSNB_info, file_CCatmo, file_CCatmo_info,
            file_reactor, file_reactor_info)
