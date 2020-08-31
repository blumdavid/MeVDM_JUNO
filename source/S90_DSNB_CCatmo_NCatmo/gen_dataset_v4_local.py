""" Script 'gen_dataset_v4_local.py (13.08.2020):

    It is used for simulation and analysis of "S90_DSNB_CCatmo_NCatmo".

    Script to generate several data-sets (virtual experiments) from the spectra,
    which are calculated with S90_DSNB_CCatmo_reactor_NCatmo_FN/gen_spectrum_v4.py:

    Script uses the function gen_dataset() from gen_dataset_v4.py,
    but is optimized to run on the local computer.

    Used spectra:
    - signal (0 for every entry)
    - DSNB
    - CCatmo on proton
    - CCatmo on C12
    - NCatmo

    Reactor and fast neutron spectrum can be neglected after PSD cut.

"""

# import of the necessary packages:
from S90_DSNB_CCatmo_NCatmo.gen_dataset_v4 import gen_dataset_v4

""" Set the DM mass in MeV (float): """
DM_mass = 0

""" Define parameters for saving of the virtual experiments: 
    number_dataset (dataset_stop - dataset_start) defines, how many datasets (virtual experiments)
    are generated. """
# INFO-me: 10000 datasets might be a good number for the analysis
# dataset_start defines the start point (integer):
dataset_start = 0
# dataset_stop defines the end point (integer):
dataset_stop = 999

""" Set boolean variable, which controls, if the txt-file with the data are saved or not: """
SAVE_DATA_TXT = True

""" Set boolean variable, which controls, if the all files (txt and png) with the data are saved or not: """
SAVE_DATA_ALL = False

""" Set boolean variable, which controls, if the data is displayed on screen or not (should be False): """
# display the simulated spectra (when True, you get a warning (can't invoke "event" command: application has been
# destroyed while executing "event generate $w <<ThemeChanged>>" (procedure "ttk::ThemeChanged" line 6) invoked from
# within "ttk::ThemeChanged") -> there is a conflict between the pyplot.close() command and the pyplot.show() command.
# BUT: datasets are generated correctly!!!
DISPLAY_DATA = False

""" set the path to the correct folder: """
# path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/1000datasets_54_4years/"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_{0}".format(DM_mass)

""" set the path of the folder, where the datasets are saved: """
path_dataset = path_output + "/datasets"

""" Files, that are read in to generate data-sets (virtual experiments): """
# path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
#             "PSD_350ns_600ns_400000atmoNC_1MeV/"

path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
            "PSD_350ns_600ns_400000atmoNC_1MeV_54_4years/"

# path_simu_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
#                "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
#                "test_350ns_600ns_400000atmoNC_1MeV/"

path_simu_NC = path_simu

file_signal = path_simu + "{0}signal_bin1000keV.txt".format(DM_mass)
file_signal_info = path_simu + "{0}signal_info_bin1000keV.txt".format(DM_mass)

file_DSNB = path_simu + "DSNB_bin1000keV_f027_PSD.txt"
file_DSNB_info = path_simu + "DSNB_info_bin1000keV_f027_PSD.txt"

file_CCatmo_p = path_simu + "CCatmo_onlyP_Osc1_bin1000keV_PSD.txt"
file_CCatmo_p_info = path_simu + "CCatmo_onlyP_info_Osc1_bin1000keV_PSD.txt"

file_CCatmo_C12 = path_simu + "CCatmo_onlyC12_Osc1_bin1000keV_PSD.txt"
file_CCatmo_C12_info = path_simu + "CCatmo_onlyC12_info_Osc1_bin1000keV_PSD.txt"

file_NCatmo = path_simu_NC + "NCatmo_onlyC12_wPSD99_bin1000keV_fit.txt"
file_NCatmo_info = path_simu_NC + "NCatmo_info_onlyC12_wPSD99_bin1000keV.txt"

""" generate the datasets with function gen_dataset() from gen_dataset_v1.py: """
gen_dataset_v4(DM_mass, SAVE_DATA_TXT, SAVE_DATA_ALL, DISPLAY_DATA, dataset_start, dataset_stop, path_output,
               path_dataset, file_signal, file_signal_info, file_DSNB, file_DSNB_info, file_CCatmo_p,
               file_CCatmo_p_info, file_NCatmo, file_NCatmo_info, file_CCatmo_C12, file_CCatmo_C12_info)
