""" script to compare the results of the Bayesian analysis for the different DM masses:

    - compare the mean of the number of background events for each DM mass




"""
import numpy as np
from matplotlib import pyplot as plt

# get the expected number of background events from the spectra:
path_expected_spectrum = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
                         "PSD_350ns_600ns_400000atmoNC_1MeV/"
path_expected_spectrum_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                            "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm" \
                            "_PSD99/test_350ns_600ns_400000atmoNC_1MeV/"

# path of the directory, where the results of the analysis are saved (string):
path_folder = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"

# expected number of background events:
B_DSNB_exp = np.sum(np.loadtxt(path_expected_spectrum + "DSNB_bin1000keV_f027_PSD.txt"))
B_CCatmo_p_exp = np.sum(np.loadtxt(path_expected_spectrum + "CCatmo_onlyP_Osc1_bin1000keV_PSD.txt"))
B_CCatmo_C12_exp = np.sum(np.loadtxt(path_expected_spectrum + "CCatmo_onlyC12_Osc1_bin1000keV_PSD.txt"))
B_NCatmo_exp = np.sum(np.loadtxt(path_expected_spectrum_NC + "NCatmo_onlyC12_wPSD99_bin1000keV_fit.txt"))

# preallocate the arrays, where the mean values are stored for each DM mass:
array_B_DSNB = []
array_B_CCatmo_p = []
array_B_CCatmo_C12 = []
array_B_reactor = []
array_B_NCatmo = []

# array of DM masses:
mass_DM = np.arange(20, 105, 5)

# loop over DM masses:
for mass in mass_DM:
    # path where the results are saved:
    path_result = path_folder + "dataset_output_{0:.0f}/result_mcmc/".format(mass)

    # load file where means are saved:
    file_result = np.loadtxt(path_result + "result_dataset_output_{0:.0f}.txt".format(mass))

    # get the mean values of the background events:
    DSNB = file_result[13]
    CCatmo_p = file_result[18]
    CCatmo_C12 = file_result[28]
    NCatmo = file_result[23]

    # append the values to the arrays:
    array_B_DSNB.append(DSNB)
    array_B_CCatmo_p.append(CCatmo_p)
    array_B_CCatmo_C12.append(CCatmo_C12)
    array_B_NCatmo.append(NCatmo)

" display results: "
plt.figure(1)
plt.hist(array_B_DSNB, bins='auto', label="DSNB")
plt.vlines(B_DSNB_exp, 0, 5)
plt.grid()
plt.legend()

plt.figure(2)
plt.hist(array_B_CCatmo_p, bins='auto', label="CCatmo p")
plt.vlines(B_CCatmo_p_exp, 0, 5)
plt.legend()
plt.grid()

plt.figure(3)
plt.hist(array_B_CCatmo_C12, bins='auto', label="CCatmo C12")
plt.vlines(B_CCatmo_C12_exp, 0, 5)
plt.legend()
plt.grid()

plt.figure(5)
plt.hist(array_B_NCatmo, bins='auto', label="NCatmo")
plt.vlines(B_NCatmo_exp, 0, 5)
plt.legend()
plt.grid()

plt.show()



























