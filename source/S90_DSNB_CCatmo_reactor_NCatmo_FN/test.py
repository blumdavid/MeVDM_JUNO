import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import sigma_ibd

input_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/" \
             "PSD_350ns_600ns_400000atmoNC_1MeV/"

input_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
           "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
           "test_350ns_600ns_400000atmoNC_1MeV/"

output_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"

DM_mass = np.arange(20, 105, 5)

CCatmo_C12 = np.loadtxt(input_path + "CCatmo_onlyC12_Osc1_bin1000keV_PSD.txt")
CCatmo_C12_new = CCatmo_C12 * 5.44
print(np.sum(CCatmo_C12_new))
# np.savetxt(output_path + "CCatmo_onlyC12_Osc1_bin1000keV_PSD.txt", CCatmo_C12_new, fmt='%1.5e')

CCatmo_p = np.loadtxt(input_path + "CCatmo_onlyP_Osc1_bin1000keV_PSD.txt")
CCatmo_p_new = CCatmo_p * 5.44
print(np.sum(CCatmo_p_new))
# np.savetxt(output_path + "CCatmo_onlyP_Osc1_bin1000keV_PSD.txt", CCatmo_p_new, fmt="%1.5e")

DSNB = np.loadtxt(input_path + "DSNB_bin1000keV_f027_PSD.txt")
DSNB_new = DSNB * 5.44
print(np.sum(DSNB_new))
# np.savetxt(output_path + "DSNB_bin1000keV_f027_PSD.txt", DSNB_new, fmt="%1.5e")

NCatmo = np.loadtxt(input_NC + "NCatmo_onlyC12_wPSD99_bin1000keV_fit.txt")
NCatmo_new = NCatmo * 5.44
print(np.sum(NCatmo_new))
# np.savetxt(output_path + "NCatmo_onlyC12_wPSD99_bin1000keV_fit.txt", NCatmo_new, fmt="%1.5e")

for mass in DM_mass:

    signal = np.loadtxt(input_path + "signal_DMmass{0:.0f}_bin1000keV_PSD.txt".format(mass))
    signal_new = signal * 5.44
    print(np.sum(signal_new))
    # np.savetxt(output_path + "signal_DMmass{0:.0f}_bin1000keV_PSD.txt".format(mass), signal_new, fmt="%1.5e")

    signal_woPSD = np.loadtxt(input_path + "signal_DMmass{0:.0f}_bin1000keV.txt".format(mass))
    signal_woPSD_new = signal_woPSD * 5.44
    print(np.sum(signal_woPSD_new))
    np.savetxt(output_path + "signal_DMmass{0:.0f}_bin1000keV.txt".format(mass), signal_woPSD_new, fmt="%1.5e")



