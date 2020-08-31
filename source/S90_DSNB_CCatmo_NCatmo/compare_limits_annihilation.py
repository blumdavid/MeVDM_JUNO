""" Script to compare the 90 % upper limits of the annihilation cross-section for different settings:

    - with datasets and without datasets

    - different prior probabilities:

        - case1: sigma = 2 * mu

        - case2:
            - sigma_DSNB = 0.4 * mu_DSNB
            - sigma_CCatmo_p = 0.25 * mu_CCatmo_p
            - sigma_CCatmo_C12 = 0.25 * mu_CCatmo_C12
            - sigma_NCatmo = 0.29 * mu_NCatmo

        - case3: sigma = 0.1 * mu

"""


import numpy as np
from matplotlib import pyplot as plt

DM_mass = np.arange(20, 105, 5)

sigma_w_1000datasets_case1 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/1000datasets/"
                                        "limit_annihilation_JUNO.txt")

sigma_w_10000datasets_case1 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                         "limit_annihilation_JUNO.txt")

sigma_wo_datasets_case1 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                     "test_Bayesian_ohne_datasets_case1/limit_annihilation_JUNO_wo_datasets.txt")

sigma_wo_datasets_case2 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                     "test_Bayesian_ohne_datasets_case2/limit_annihilation_JUNO_wo_datasets.txt")

sigma_wo_datasets_case3 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                     "test_Bayesian_ohne_datasets_case3/limit_annihilation_JUNO_wo_datasets.txt")

sigma_wo_datasets_5timesJUNO_case3 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                                "test_Bayesian_ohne_datasets_case3_5timesJUNO/"
                                                "limit_annihilation_JUNO_wo_datasets.txt")

sigma_wo_datasets_10precentJUNO_case2 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                                   "test_Bayesian_ohne_datasets_case2_10precentJUNO/"
                                                   "limit_annihilation_JUNO_wo_datasets.txt")

sigma_w_1000datasets_54_4years_case1 = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_NCatmo/"
                                                  "1000datasets_54_4years/limit_annihilation_JUNO.txt")

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

plt.figure(1)
plt.semilogy(DM_mass, sigma_w_1000datasets_case1, "b-", label="with 1000 datasets (case1)")
plt.semilogy(DM_mass, sigma_w_10000datasets_case1, "k", label="with 10,000 datasets (case1)")
plt.semilogy(DM_mass, sigma_wo_datasets_case1, "r-", label="without datasets (case1)")
# plt.semilogy(DM_mass, sigma_wo_datasets_case2, "r--", label="without datasets (case2)")
# plt.semilogy(DM_mass, sigma_wo_datasets_case3, "r:", label="without datasets (case3)")
# plt.semilogy(DM_mass, sigma_wo_datasets_5timesJUNO_case3, "g:", label="without datasets (case3, 5 times JUNO)")
# plt.semilogy(DM_mass, sigma_wo_datasets_10precentJUNO_case2, "g--", label="without datasets (case2, 1 year JUNO)")
plt.semilogy(DM_mass, sigma_w_1000datasets_54_4years_case1, "g-", label="with 1000 datasets (case1, 54.4 years JUNO)")
plt.semilogy(DM_mass_HyperK, sigma_anni_HyperK, linestyle=":", color='black', linewidth=2.0,
             label="90% C.L. limit simulated for Hyper-K (10 years)")
plt.xlabel("DM mass in MeV")
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$")
plt.grid()
plt.legend()

plt.show()

