""" script to calculate the 90 % upper limit of the number of signal events and the 90 % upper limit of the
    DM annihilation cross-section with the ansatz described in these two papers:

    - 0710.5420: old Super-K limit

    - Phys.Rev D 97, 075039 (2018): new Super-K limit

"""
import numpy as np
import scipy.optimize as op
from matplotlib import pyplot as plt
from matplotlib import rcParams
from gen_spectrum_functions import limit_annihilation_crosssection_v2


def chi_squared(param, f_s, f_bkg_1, f_bkg_2, f_bkg_3, f_bkg_4, f_bkg_5, array_n, sigma_stat, sigma_sys):
    """
    Chi squared function with 6 free parameter: number of signal events and number of bkg events of the 5 backgrounds

    :param param: number of signal events, number of bkg1 events, number of bkg2 events, number of bkg3 events,
    number of bkg4 events, number of bkg5 events
    :param f_s: spectral shape of the signal contribution (array)
    :param f_bkg_1: spectral shape of bkg1 per bin (array)
    :param f_bkg_2: spectral shape of bkg2 per bin (array)
    :param f_bkg_3: spectral shape of bkg3 per bin (array)
    :param f_bkg_4: spectral shape of bkg4 per bin (array)
    :param f_bkg_5: spectral shape of bkg5 per bin (array)
    :param array_n: observed number of events (array)
    :param sigma_stat: statistical error in %
    :param sigma_sys: systematic error in %
    :return:
    """
    # get free parameters from param (number of signal, bkg1, bkg2, bkg3, bkg4, bkg5 events):
    number_s, number_bkg_1, number_bkg_2, number_bkg_3, number_bkg_4, number_bkg_5 = param

    # set statistical and systematic error (not in %):
    sigma_stat = sigma_stat / 100.0
    sigma_sys = sigma_sys / 100.0

    # calculate summand of the chi^2 function:
    summand = (number_s * f_s + number_bkg_1 * f_bkg_1 + number_bkg_2 * f_bkg_2 + number_bkg_3 * f_bkg_3 +
               number_bkg_4 * f_bkg_4 + number_bkg_5 * f_bkg_5 - array_n)**2 / (sigma_stat**2 + sigma_sys**2)

    # chi squared:
    chi_2 = np.sum(summand)

    return chi_2


def chi_squared_alpha(param, number_s, f_s, f_bkg_1, f_bkg_2, f_bkg_3, f_bkg_4, f_bkg_5, array_n, sigma_stat,
                      sigma_sys):
    """
    Chi squared function with 4 free parameter: number of bkg events of the 4 backgrounds

    The number of signal events is fixed.

    :param param: number of bkg1 events, number of bkg2 events, number of bkg3 events, number of bkg4 events,
    number of bkg5 events
    :param number_s: number of signal events, fix parameter
    :param f_s: spectral shape of the signal contribution (array)
    :param f_bkg_1: spectral shape of bkg1 per bin (array)
    :param f_bkg_2: spectral shape of bkg2 per bin (array)
    :param f_bkg_3: spectral shape of bkg3 per bin (array)
    :param f_bkg_4: spectral shape of bkg4 per bin (array)
    :param f_bkg_5: spectral shape of bkg5 per bin (array)
    :param array_n: observed number of events (array)
    :param sigma_stat: statistical error in %
    :param sigma_sys: systematic error in %
    :return:
    """
    # get free parameters from param (number of bkg1, bkg2, bkg3, bkg4, bkg5 events):
    number_bkg_1, number_bkg_2, number_bkg_3, number_bkg_4, number_bkg_5 = param

    # set statistical and systematic error (not in %):
    sigma_stat = sigma_stat / 100.0
    sigma_sys = sigma_sys / 100.0

    # calculate summand of the chi^2 function:
    summand = (number_s * f_s + number_bkg_1 * f_bkg_1 + number_bkg_2 * f_bkg_2 + number_bkg_3 * f_bkg_3 +
               number_bkg_4 * f_bkg_4 + number_bkg_5 * f_bkg_5 - array_n) ** 2 / (sigma_stat ** 2 + sigma_sys ** 2)

    # chi squared:
    chi_2 = np.sum(summand)

    return chi_2


""" load bkg spectra: """
input_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"
# DSNB background in events pre bin:
DSNB = input_path + "DSNB_bin500keV_PSD.txt"
DSNB = np.loadtxt(DSNB)
# expected number of DSNB events:
N_DSNB_true = np.sum(DSNB)
# spectral shape of DSNB per bin:
f_DSNB = DSNB / N_DSNB_true

# reactor background in events pre bin:
reactor = input_path + "Reactor_NH_power36_bin500keV_PSD.txt"
reactor = np.loadtxt(reactor)
# expected number of reactor events:
N_reactor_true = np.sum(reactor)
# spectral shape of reactor background per bin:
f_reactor = reactor / N_reactor_true

# CCatmo background on protons in events pre bin:
CCatmo_p = input_path + "CCatmo_onlyP_Osc1_bin500keV_PSD.txt"
CCatmo_p = np.loadtxt(CCatmo_p)
# expected number of CCatmo_p events:
N_CCatmo_p_true = np.sum(CCatmo_p)
# spectral shape of CCatmo background on protons per bin:
f_CCatmo_p = CCatmo_p / N_CCatmo_p_true

# CCatmo background on C12 in events pre bin:
CCatmo_C12 = input_path + "CCatmo_onlyC12_Osc1_bin500keV_PSD.txt"
CCatmo_C12 = np.loadtxt(CCatmo_C12)
# expected number of CCatmo_C12 events:
N_CCatmo_C12_true = np.sum(CCatmo_C12)
# spectral shape of CCatmo background on C12 per bin:
f_CCatmo_C12 = CCatmo_C12 / N_CCatmo_C12_true

# NC atmo background in events pre bin
input_path_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
                "test_10to20_20to30_30to40_40to100_final/"
NCatmo = input_path_NC + "NCatmo_onlyC12_wPSD99_bin500keV.txt"
NCatmo = np.loadtxt(NCatmo)
# expected number of NC atmo events:
N_NCatmo_true = np.sum(NCatmo)
# spectral shape of NCatmo background per bin:
f_NCatmo = NCatmo / N_NCatmo_true

""" set statistical and systematical error in %: """
Sigma_stat = 50.0
Sigma_sys = 50.0

# set DM masses in MeV:
DM_mass = np.arange(15, 105, 5)
# DM_mass = [15]

# preallocate array, where the 90 % upper limit of S is stored:
S_90_old_array = []
S_90_new_array = []
# preallocate array, where PSD efficiency for each DM mass is stored:
PSD_eff_array = []

# loop over DM masses:
for mass in DM_mass:
    print("----------------------------------------------------------------------------")
    print("\nDM mass = {0:d}".format(mass))

    # signal spectrum in events per bin
    signal = input_path + "signal_DMmass{0:d}_bin500keV_PSD.txt".format(mass)
    signal = np.loadtxt(signal)
    # expected number of signal events (after PSD):
    N_signal_true = np.sum(signal)
    # normalize to 1 to get spectral shape:
    f_signal = signal / N_signal_true

    # number of signal events before PSD:
    N_signal_wo_PSD = np.sum(np.loadtxt(input_path + "signal_DMmass{0}_bin500keV.txt".format(mass)))
    # PSD efficiency of this DM mass:
    PSD_efficiency = float(N_signal_true) / float(N_signal_wo_PSD)
    # append to array:
    PSD_eff_array.append(PSD_efficiency)

    """ Set array with observed number of events (measured data): """
    # background-only spectrum:
    Data = DSNB + CCatmo_p + CCatmo_C12 + reactor + NCatmo

    """ minimize chi-squared function to get best fit values for number of signal and background: """
    # guess of the parameters (as guess, the expected number of events from the simulated spectrum are used)
    # (np.array of float):
    parameter_guess = np.array([N_signal_true, N_DSNB_true, N_CCatmo_p_true, N_CCatmo_C12_true, N_reactor_true,
                                N_NCatmo_true])

    # bounds of the parameters (parameters have to be positive or zero) (tuple):
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # Minimize chi_squared function with the L-BFGS-B method for the defined bounds above:
    result_chi_squared = op.minimize(chi_squared, parameter_guess, args=(f_signal, f_DSNB, f_CCatmo_p, f_CCatmo_C12,
                                                                         f_reactor, f_NCatmo, Data, Sigma_stat,
                                                                         Sigma_sys),
                                     method='L-BFGS-B', bounds=bnds, options={'disp': None})

    # get the best-fit parameters from the minimization (float):
    N_signal_bf, N_DSNB_bf, N_CCatmo_p_bf, N_CCatmo_C12_bf, N_reactor_bf, N_NCatmo_bf = result_chi_squared["x"]

    # calculate value of chi_squared() for the best fit parameters:
    value_chi_squared_bestfit = chi_squared(result_chi_squared["x"], f_signal, f_DSNB, f_CCatmo_p, f_CCatmo_C12,
                                            f_reactor, f_NCatmo, Data, Sigma_stat, Sigma_sys)

    print("\nbest fit parameters:")
    print("signal = {0:.3f}".format(N_signal_bf))
    print("DSNB = {0:.3f}".format(N_DSNB_bf))
    print("CCatmo on p = {0:.3f}".format(N_CCatmo_p_bf))
    print("CCatmo on C12 = {0:.3f}".format(N_CCatmo_C12_bf))
    print("reactor = {0:.3f}".format(N_reactor_bf))
    print("NCatmo = {0:.3f}".format(N_NCatmo_bf))
    print("value of chi_squared corresponding to best fit parameter = {0:.5f}".format(value_chi_squared_bestfit))

    """ Minimize chi_squared_alpha for each fixed value of S with free parameter b1, b2, b3, b4, b5:
        'The limit is obtained by increasing the value of α and evaluating the χ2 function using b1, b2, b3, b4 
        and b5 as free parameters.'
    """
    # set array of number of signal events:
    array_number_S = np.arange(0, 100, 0.01)

    # preallocate arrays:
    Chi_squared_S = []
    # preallocate arrays, where the fit values of the backgrounds are stored:
    dsnb_array = []
    ccatmo_p_array = []
    ccatmo_C12_array = []
    reactor_array = []
    ncatmo_array = []

    # parameter guess for the number of background events:
    parameter_guess_1 = np.array([N_DSNB_bf, N_CCatmo_p_bf, N_CCatmo_C12_bf, N_reactor_bf, N_NCatmo_bf])

    # bounds of the parameters (parameters have to be positive or zero) (tuple):
    bnds_1 = ((0, None), (0, None), (0, None), (0, None), (0, None))

    # loop over array_number_S:
    for index in range(len(array_number_S)):
        # minimize chi_squared_alpha for each value of array_number_S[index]:
        result_chi_squared_s = op.minimize(chi_squared_alpha, parameter_guess_1,
                                           args=(array_number_S[index], f_signal, f_DSNB, f_CCatmo_p, f_CCatmo_C12,
                                                 f_reactor, f_NCatmo, Data, Sigma_stat, Sigma_sys),
                                           method='L-BFGS-B', bounds=bnds_1, options={'disp': None})

        # append value of chi_squared_s to array:
        Chi_squared_S.append(chi_squared_alpha(result_chi_squared_s["x"], array_number_S[index], f_signal, f_DSNB,
                                               f_CCatmo_p, f_CCatmo_C12, f_reactor, f_NCatmo, Data, Sigma_stat,
                                               Sigma_sys))
        # get fit value of the backgrounds corresponding to array_number_S[index]:
        n_dsnb, n_ccatmo_p, n_ccatmo_c12, n_reactor, n_ncatmo = result_chi_squared_s["x"]
        # append values to array:
        dsnb_array.append(n_dsnb)
        ccatmo_p_array.append(n_ccatmo_p)
        ccatmo_C12_array.append(n_ccatmo_c12)
        reactor_array.append(n_reactor)
        ncatmo_array.append(n_ncatmo)

    # convert Chi_squared_S to numpy array:
    Chi_squared_S = np.asarray(Chi_squared_S)

    # plt.plot(array_number_S, Chi_squared_S, "r", label="Chi^2")
    # plt.plot(array_number_S, np.exp(-Chi_squared_S), "b", label="exp(-Chi^2)")
    # plt.legend()
    # plt.show()

    """ calculate limit with ansatz of old Super-K paper: """
    # calculate normalization factor K:
    factor_K_old = 1.0 / np.sum(np.exp(-Chi_squared_S))

    # calculate relative probability:
    P_old = factor_K_old * np.exp(-Chi_squared_S)

    # preallocate values of the sum of P and the index of P:
    sum_P_old = 0
    index_P_old = 0

    while sum_P_old <= 0.9:
        # as long as the sum of P is smaller than 0.9, sum up the next value of P:
        index_P_old += 1
        sum_P_old = np.sum(P_old[0:index_P_old])

    # get 90 % upper limit of the number of signal events:
    S_90_old = array_number_S[index_P_old]
    # append it to array:
    S_90_old_array.append(S_90_old)

    """ calculate limit with ansatz of new Super-K paper: """
    # calculate normalization factor K:
    factor_K_new = 1.0 / np.sum(Chi_squared_S)

    # calculate quotient:
    P_new = factor_K_new * Chi_squared_S

    # preallocate values of the sum of P and the index of P:
    sum_P_new = 0
    index_P_new = 0

    while sum_P_new <= 0.9:
        # as long as the sum of P is smaller than 0.9, sum up the next value of P:
        index_P_new += 1
        sum_P_new = np.sum(P_new[0:index_P_new])

    # get 90 % upper limit of the number of signal events:
    S_90_new = array_number_S[index_P_new]
    # append it to array:
    S_90_new_array.append(S_90_new)

    print("\nAnsatz of old Super-K limit:")
    print("\nS_90 = {0:.5f}".format(S_90_old))
    print("number of DSNB = {0:.3f}".format(dsnb_array[index_P_old]))
    print("number of CCatmo on p = {0:.3f}".format(ccatmo_p_array[index_P_old]))
    print("number of CCatmo on C12 = {0:.3f}".format(ccatmo_C12_array[index_P_old]))
    print("number of reactor = {0:.3f}".format(reactor_array[index_P_old]))
    print("number of NCatmo = {0:.3f}".format(ncatmo_array[index_P_old]))
    print("corresponding chi^2 = {0:.5f}".format(Chi_squared_S[index_P_old]))

    print("\nAnsatz of new Super-K limit:")
    print("\nS_90 = {0:.5f}".format(S_90_new))
    print("number of DSNB = {0:.3f}".format(dsnb_array[index_P_new]))
    print("number of CCatmo on p = {0:.3f}".format(ccatmo_p_array[index_P_new]))
    print("number of CCatmo on C12 = {0:.3f}".format(ccatmo_C12_array[index_P_new]))
    print("number of reactor = {0:.3f}".format(reactor_array[index_P_new]))
    print("number of NCatmo = {0:.3f}".format(ncatmo_array[index_P_new]))
    print("corresponding chi^2 = {0:.5f}".format(Chi_squared_S[index_P_new]))

""" 90 % upper limit of annihilation cross-section: """
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
time_year = 10
time_s = time_year * 3.156 * 10 ** 7
epsilon_IBD = 0.67005
epsilon_mu_veto = 0.9717
sigma_anni_natural = 3 * 10 ** (-26)

limit_anni_90_old_array = []
limit_anni_90_new_array = []

for index in range(len(DM_mass)):
    # ansatz of old limit:
    limit_anni_90_old = limit_annihilation_crosssection_v2(S_90_old_array[index], DM_mass[index], J_avg, N_target,
                                                           time_s, epsilon_IBD, epsilon_mu_veto, PSD_eff_array[index],
                                                           MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_anni_90_old_array.append(limit_anni_90_old)

    # ansatz of new limit:
    limit_anni_90_new = limit_annihilation_crosssection_v2(S_90_new_array[index], DM_mass[index], J_avg, N_target,
                                                           time_s, epsilon_IBD, epsilon_mu_veto, PSD_eff_array[index],
                                                           MASS_NEUTRON, MASS_PROTON, MASS_POSITRON)
    limit_anni_90_new_array.append(limit_anni_90_new)

limit_anni_90_old_array = np.asarray(limit_anni_90_old_array)
limit_anni_90_new_array = np.asarray(limit_anni_90_new_array)

""" Plot of the 90% upper limit of the number of signal events from JUNO for old and new Ansatz: """
# increase distance between plot and title:
rcParams["axes.titlepad"] = 20

h1 = plt.figure(1, figsize=(10, 6))
plt.plot(DM_mass, S_90_old_array, marker='x', markersize='6.0', linestyle='-', color='red', linewidth=2.0,
         label='90% upper limit $S_{90}$ (old Super-K approach)')
plt.plot(DM_mass, S_90_new_array, marker='x', markersize='6.0', linestyle='-', color='blue', linewidth=2.0,
         label='90% upper limit $S_{90}$ (new Super-K approach)')
plt.xlim(10, np.max(DM_mass))
# plt.ylim(0, 10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$S_{90}$", fontsize=13)
plt.title("90% upper probability limit on the number of signal events from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()

""" Semi-logarithmic plot of the 90% upper limit of the DM self-annihilation cross-section from JUNO: """
# load the result of the 90 % limit of the annihilation cross-section of the Bayesian ansatz with 1000 datasets:
limit_anni_Bayesian_w_datasets = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo/"
                                            "limit_annihilation_JUNO.txt")
# load the result of the 90 % limit of the annihilation cross-section of the Bayesian ansatz without datasets:
limit_anni_Bayesian_wo_datasets = np.loadtxt("/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo/"
                                             "test_Bayesian_ohne_datasets/limit_annihilation_JUNO_wo_datasets.txt")

# maximum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_max = 10 ** (-22)
# minimum value for the 90% limit on the annihilation cross-section in cm**3/s (float):
y_min = 5 * 10 ** (-27)

h2 = plt.figure(2, figsize=(10, 6))
plt.semilogy(DM_mass, limit_anni_90_old_array, marker='x', markersize='6.0', linestyle='-', color='green',
             linewidth=2.0, label='Approach of 0710.5420')
plt.semilogy(DM_mass, limit_anni_90_new_array, marker='x', markersize='6.0', linestyle='-', color='blue', linewidth=2.0,
             label='Approach of Phys.Rev D 97, 075039')
plt.semilogy(DM_mass, limit_anni_Bayesian_w_datasets, marker='x', markersize='6.0', linestyle='-', color='red',
             linewidth=2.0, label='Bayesian approach (1000 datasets)')
plt.semilogy(DM_mass, limit_anni_Bayesian_wo_datasets, marker='x', markersize='6.0', linestyle='--', color='red',
             linewidth=2.0, label='Bayesian approach (without datasets)')
plt.axhline(sigma_anni_natural, linestyle=':', color='black',
            label='natural scale of the annihilation cross-section\n($<\\sigma_A v>_{natural}=3*10^{-26}\,cm^3/s$)')
plt.fill_between(DM_mass, y_min, sigma_anni_natural, facecolor="grey", alpha=0.25, hatch='/')
plt.xlim(np.min(DM_mass), np.max(DM_mass))
plt.ylim(y_min, y_max)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dark Matter mass in MeV", fontsize=13)
plt.ylabel("$<\\sigma_A v>_{90}$ in $cm^3/s$", fontsize=13)
plt.title("90% upper limit on the total DM self-annihilation cross-section from the JUNO experiment", fontsize=15)
plt.legend(fontsize=12)
plt.grid()

plt.show()
