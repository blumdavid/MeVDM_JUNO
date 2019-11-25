""" Script 'gen_dataset_v2_local.py (29.10.2019):

    It is used for simulation and analysis of "S90_DSNB_CCatmo_reactor_NCatmo_FN".

    Script to generate several data-sets (virtual experiments) from the spectra,
    which are calculated with gen_spectrum_v3.py:

    Script uses the function gen_dataset() from gen_dataset_v2.py,
    but is optimized to run on the local computer.

"""

# import of the necessary packages:
from work.MeVDM_JUNO.source.S90_DSNB_CCatmo_reactor_NCatmo_FN.gen_dataset_v2 import gen_dataset_v2

""" Set the DM mass in MeV (float): """
DM_mass = 0

""" Define parameters for saving of the virtual experiments: 
    number_dataset (dataset_stop - dataset_start) defines, how many datasets (virtual experiments)
    are generated. """
# INFO-me: 10000 datasets might be a good number for the analysis
# dataset_start defines the start point (integer):
dataset_start = 0
# dataset_stop defines the end point (integer):
dataset_stop = 9

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
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN"

""" set the path of the output folder: """
path_output = path_folder + "/dataset_output_{0}".format(DM_mass)

""" set the path of the folder, where the datasets are saved: """
path_dataset = path_output + "/datasets"

""" Files, that are read in to generate data-sets (virtual experiments): """
path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v2/S90_DSNB_CCatmo_reactor_NCatmo_FN"
path_simu_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
               "DCR_results_16000mm_10MeVto100MeV_500nsto1ms_mult1_2400PEto3400PE_dist500mm_R17700mm_PSD99/"
file_signal = path_simu + "/{0}signal_bin500keV.txt".format(DM_mass)
file_signal_info = path_simu + "/{0}signal_info_bin500keV.txt".format(DM_mass)
file_DSNB = path_simu + "/DSNB_EmeanNuXbar22_bin500keV_PSD.txt"
file_DSNB_info = path_simu + "/DSNB_info_EmeanNuXbar22_bin500keV_PSD.txt"
file_CCatmo = path_simu + "/CCatmo_total_Osc1_bin500keV_PSD.txt"
file_CCatmo_info = path_simu + "/CCatmo_total_info_Osc1_bin500keV_PSD.txt"
file_reactor = path_simu + "/Reactor_NH_power36_bin500keV_PSD.txt"
file_reactor_info = path_simu + "/Reactor_info_NH_power36_bin500keV_PSD.txt"
file_NCatmo = path_simu_NC + "/NCatmo_onlyC12_wPSD99_bin500keV.txt"
file_NCatmo_info = path_simu_NC + "/NCatmo_info_onlyC12_wPSD99_bin500keV.txt"
file_fastN = path_simu + "/fast_neutron_33events_bin500keV_PSD.txt"
file_fastN_info = path_simu + "/fast_neutron_info_33events_bin500keV_PSD.txt"

""" generate the datasets with function gen_dataset() from gen_dataset_v1.py: """
gen_dataset_v2(DM_mass, SAVE_DATA_TXT, SAVE_DATA_ALL, DISPLAY_DATA, dataset_start, dataset_stop, path_output,
               path_dataset, file_signal, file_signal_info, file_DSNB, file_DSNB_info, file_CCatmo, file_CCatmo_info,
               file_reactor, file_reactor_info, file_NCatmo, file_NCatmo_info, file_fastN, file_fastN_info)
