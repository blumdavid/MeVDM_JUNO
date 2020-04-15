""" test """

import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import limit_annihilation_crosssection


def chi_squared(number_s, f_s, array_bkg):
    """

    :param number_s: number of signal events (free parameter)
    :param f_s: spectral shape of the signal contribution (array)
    :param array_bkg: array of the number of background events per bin
    :return:
    """
    # array of the total number of events (signal + bkg) per bin as function of number_s:
    array_n = number_s * f_s + array_bkg

    logarith = np.log(array_n / array_bkg)

    # calculate chi-squared function:
    chi_2 = 2 * np.sum(array_n * logarith + array_bkg - array_n)

    return chi_2


# load bkg spectra:
input_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"
DSNB = input_path + "DSNB_bin500keV_PSD.txt"
DSNB = np.loadtxt(DSNB)
reactor = input_path + "Reactor_NH_power36_bin500keV_PSD.txt"
reactor = np.loadtxt(reactor)
CCatmo = input_path + "CCatmo_total_Osc1_bin500keV_PSD.txt"
CCatmo = np.loadtxt(CCatmo)

input_path_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
                "test_10to20_20to30_30to40_40to100_final/"
NCatmo = input_path_NC + "NCatmo_onlyC12_wPSD99_bin500keV.txt"
NCatmo = np.loadtxt(NCatmo)

# calculate total background spectrum:
background = DSNB + reactor + CCatmo + NCatmo

# set DM masses in MeV:
DM_mass = np.arange(15, 105, 5)
# DM_mass = [50]

# preallocate array, where the 90 % upper limit of S is stored:
S_90_array = []

# loop over DM masses:
for mass in DM_mass:

    signal = input_path + "signal_DMmass{0:d}_bin500keV_PSD.txt".format(mass)
    signal = np.loadtxt(signal)
    # normalize signal to 1 to get spectral shape:
    signal = signal / np.sum(signal)

    # set array of number of signal events:
    array_number_S = np.arange(0, 10, 0.00001)

    # preallocate arrays:
    Chi_squared = []
    array_index_90 = []

    # calculate chi square function:
    for index in range(len(array_number_S)):
        Chi_2 = chi_squared(array_number_S[index], signal, background)
        Chi_squared.append(Chi_2)

        if 0.89999 < Chi_2 < 0.90001:
            array_index_90.append(index)

    # wie Li-Cheng:
    
    # preallocate array, where absolute difference to 0.9 is stored:
    array_diff = []

    # find entry of array_index_90, where Chi_squared is nearest to 0.9:
    for index in range(len(array_index_90)):
        # calculate difference to 0.9 and take absolute value:
        diff = np.abs(0.9 - Chi_squared[array_index_90[index]])
        # append to array:
        array_diff.append(diff)

    # get index, where difference is the smallest:
    index_min = np.where(array_diff == np.amin(array_diff))[0][0]

    # get value of array_number_S corresponding to index_min:
    S_90 = array_number_S[array_index_90[index_min]]
    
    # consider degree of freedom:
    S_90 = S_90

    # append S_90 to array:
    S_90_array.append(S_90)

    print("DM mass = {0:d}".format(mass))
    print("S_90 =")
    print(S_90)
    # print("chi^2 = {0:.5f}".format(Chi_squared[array_index_90[index_min]]))

plt.plot(DM_mass, S_90_array)
plt.show()

# Calculate the 90 percent probability limit of the averaged DM self-annihilation cross-section for DM with
# mass of "mass" MeV in cm**2 (float)
J_avg = 5
# mass of positron in MeV (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (float constant):
MASS_NEUTRON = 939.56536

N_target = 1.450000000e+33
time_s = 10 * 3.156 * 10 ** 7
epsilon_IBD = 0.67005
sigma_anni_natural = 3 * 10**(-26)

limit_90_array = []

for index in range(len(DM_mass)):
    limit_90 = limit_annihilation_crosssection(S_90_array[index], DM_mass[index], J_avg, N_target, time_s, epsilon_IBD,
                                               MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_90_array.append(limit_90)

# consider the degree of freedom:
limit_90_array = np.asarray(limit_90_array)

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way, obtained from Super-K Data.
    New Super-K limit of 2853 days = 7.82 years of data (the old limit from 0710.5420 was for only 1496 days of data).
    (they have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 4 of the paper 
    'Dark matter-neutrino interactions through the lens of their cosmological implications', PhysRevD.97.075039)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_new.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_SuperK = np.array([10.5384, 10.9966, 10.9966, 11.2257, 11.9129, 12.8293, 14.433, 15.1203, 16.2658, 16.953,
                           17.4112, 18.0985, 18.5567, 19.244, 20.3895, 21.5349, 22.9095, 23.8259, 24.9714, 25.8877,
                           27.2623, 28.1787, 28.866, 30.2405, 32.5315, 33.9061, 35.7388, 37.5716, 39.4044, 41.4662,
                           43.5281, 45.3608, 47.1936, 49.2554, 50.4009, 51.7755, 53.3792, 54.9828, 57.9611, 60.9393,
                           62.543, 64.3757, 66.8958, 69.874, 72.6231, 75.3723, 78.5796, 81.7869, 85.2234, 88.6598,
                           91.4089, 94.1581, 98.5109, 100.802])
# 90% limit of the self-annihilation cross-section in cm^3/s (np.array of float):
sigma_anni_SuperK = np.array([2.50761E-22, 2.20703E-22, 1.90157E-22, 1.38191E-22, 4.19682E-23, 8.15221E-24, 1.5502E-24,
                              8.18698E-25, 4.32374E-25, 3.3493E-25, 3.01122E-25, 2.70727E-25, 2.59447E-25, 2.59447E-25,
                              2.76551E-25, 2.94782E-25, 3.01122E-25, 2.88575E-25, 2.70727E-25, 2.53984E-25, 2.48636E-25,
                              2.53984E-25, 2.70727E-25, 3.14215E-25, 4.51174E-25, 5.23647E-25, 5.82438E-25, 6.07762E-25,
                              5.94965E-25, 5.94965E-25, 6.34188E-25, 7.51892E-25, 9.30201E-25, 1.10284E-24, 1.17555E-24,
                              1.17555E-24, 1.1508E-24, 1.07962E-24, 9.10614E-25, 7.51892E-25, 6.90537E-25, 6.61763E-25,
                              6.75997E-25, 7.3606E-25, 8.36308E-25, 9.10614E-25, 9.30201E-25, 8.91441E-25, 8.54296E-25,
                              8.7267E-25, 9.70646E-25, 1.1508E-24, 1.72424E-24, 2.22589E-24])

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way expected for 
    Hyper-Kamiokande after 10 years.
    (they have used the canonical value J_avg = 5, the results are digitized from figure 1, on page 2 of the paper 
    'Implications of a Dark Matter-Neutrino Coupling at Hyper–Kamiokande', Arxiv:1805.09830)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/HyperK_limit_no_Gd.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_HyperK = np.array([11.4202, 11.4898, 11.5711, 11.6313, 11.6313, 11.7241, 11.7859, 11.9793, 12.0118, 12.1336,
                           12.2554, 12.4642, 12.4642, 12.6034, 12.7426, 12.7774, 12.8795, 13.0142, 13.0906, 13.2298,
                           13.4386, 13.6473, 13.7865, 13.9953, 14.1323, 14.2737, 14.4129, 14.5544, 14.6936, 14.7771,
                           14.9719, 15.1785, 15.4337, 15.6657, 16.5727, 17.1665, 17.8641, 19.0992, 20.8513, 22.2884,
                           23.9136, 25.4049, 26.8199, 28.4739, 30.005, 31.5361, 33.0673, 34.5984, 36.1922, 37.6607,
                           39.1919, 40.723, 42.3168, 43.8201, 45.3164, 46.8476, 48.4251, 49.7794, 52.0071, 54.2249,
                           55.7561, 57.1908, 59.418, 60.9394, 62.4374, 63.9686, 65.4997, 67.0309, 68.562, 70.0931,
                           71.6243, 73.1554, 74.7534, 76.281, 77.782, 79.3132, 80.8443, 82.3423, 83.8734, 85.4046,
                           86.9357, 88.4234, 89.998, 91.5292, 93.123, 94.5915, 96.0961, 97.6537, 99.1849, 100.716])

# 90 % limit of the self-annihilation cross-section in cm^3/s (array of float):
sigma_anni_HyperK = np.array([2.81969E-23, 2.3984E-23, 2.07369E-23, 1.79792E-23, 1.5389E-23, 1.34158E-23, 1.08796E-23,
                              9.07639E-24, 7.67308E-24, 6.07612E-24, 5.10954E-24, 4.29904E-24, 3.72832E-24, 3.1529E-24,
                              2.75079E-24, 2.42671E-24, 2.07014E-24, 1.76387E-24, 1.53958E-24, 1.34482E-24,
                              1.07845E-24, 8.76268E-25, 7.47593E-25, 6.49867E-25, 5.60194E-25, 4.82895E-25,
                              4.18958E-25, 3.64793E-25, 3.08801E-25, 2.6732E-25, 2.2808E-25, 2.00699E-25, 1.5469E-25,
                              1.32772E-25, 6.85055E-26, 5.56656E-26, 4.55841E-26, 3.74336E-26, 3.91269E-26,
                              4.31199E-26, 4.78795E-26, 5.27168E-26, 5.75659E-26, 6.33309E-26, 6.96262E-26,
                              7.63859E-26, 8.27672E-26, 8.99002E-26, 9.75563E-26, 1.05156E-25, 1.12134E-25, 1.207E-25,
                              1.28126E-25, 1.34863E-25, 1.40667E-25, 1.42365E-25, 1.41514E-25, 1.35923E-25,
                              1.23903E-25, 1.07922E-25, 9.82287E-26, 8.90453E-26, 8.01206E-26, 7.52633E-26,
                              7.20222E-26, 7.08082E-26, 7.08747E-26, 7.22607E-26, 7.50706E-26, 7.95367E-26,
                              8.48912E-26, 9.1121E-26, 1.00294E-25, 1.12329E-25, 1.24761E-25, 1.38262E-25,
                              1.54204E-25, 1.7043E-25, 1.80407E-25, 1.87869E-25, 1.90304E-25, 1.90679E-25, 1.92784E-25,
                              1.96469E-25, 2.02367E-25, 2.13703E-25, 2.27027E-25, 2.43049E-25, 2.62985E-25,
                              2.81874E-25])

""" 90% C.L. bound on the total DM self-annihilation cross-section from the whole Milky Way obtained from KamLAND data 
    for 2343 days = 6.42 years of data taking.
    (they have used the canonical value J_avg = 5, the results are digitized from figure 5, on page 19 of the paper 
    'Search for extraterrestrial antineutrino sources with the KamLAND detector', Arxiv:1105.3516)
    The digitized data is saved in "/home/astro/blum/PhD/paper/KamLand/limit_KamLAND.csv".
"""
# Dark matter mass in MeV (array of float):
DM_mass_Kamland = np.array([10.321, 11.8765, 12.4691, 13.284, 13.8025, 14.2469, 14.7654, 15.358, 15.358, 15.6543,
                            15.9506, 16.6173, 17.8025, 18.4691, 19.2099, 19.9506, 20.6173, 21.358, 22.0988, 22.3951,
                            22.6173, 23.0617, 23.4321, 23.9506, 24.6914, 25.4321, 26.0988, 26.8395, 27.5802, 28.321,
                            29.8765])

# 90 % limit of the self-annihilation cross-section in cm^3/s (array of float):
sigma_anni_Kamland = np.array([1.18275E-24, 9.19504E-25, 1.34141E-24, 1.95691E-24, 2.21943E-24, 2.36361E-24,
                               2.29039E-24, 1.89629E-24, 2.01948E-24, 1.89629E-24, 1.72545E-24, 1.42856E-24,
                               1.04285E-24, 9.48901E-25, 8.91017E-25, 9.79238E-25, 1.22056E-24, 1.52136E-24,
                               1.72545E-24, 1.67199E-24, 1.57E-24, 1.29986E-24, 1.04285E-24, 9.19504E-25, 8.91017E-25,
                               9.79238E-25, 1.1106E-24, 1.18275E-24, 1.14611E-24, 9.79238E-25, 3.55838E-24])

""" Semi-log. plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO, Super-K, Hyper-K and 
KamLAND: """
# maximum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_max = 10 ** (-22)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 10 ** (-26)

h3 = plt.figure(3, figsize=(13, 8))
plt.semilogy(DM_mass, limit_90_array, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
             label='90% C.L. limit simulated for JUNO (10 years)')
plt.fill_between(DM_mass, limit_90_array, y_max, facecolor='red', alpha=0.4)
plt.semilogy(DM_mass_SuperK, sigma_anni_SuperK, linestyle="--", color='black', linewidth=2.0,
             label="90% C.L. limit from Super-K data (7.82 years)")
plt.semilogy(DM_mass_HyperK, sigma_anni_HyperK, linestyle=":", color='black', linewidth=2.0,
             label="90% C.L. limit simulated for Hyper-K (10 years)")
plt.semilogy(DM_mass_Kamland, sigma_anni_Kamland, linestyle="-", color='black', linewidth=2.0,
             label="90% C.L. limit from KamLAND data (6.42 years)")
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section ($<\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass_SuperK, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass_Kamland), np.max(DM_mass_SuperK))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=15)
plt.ylabel("$<\sigma_A v>_{90}$ in $cm^3/s$", fontsize=15)
plt.title("90% upper limit on the total DM self-annihilation cross-section \nfrom the JUNO experiment", fontsize=20)
plt.legend(fontsize=13)
plt.grid()
plt.show()








