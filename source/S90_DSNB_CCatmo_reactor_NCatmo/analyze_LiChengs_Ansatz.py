""" script to use the ansatz of Li-Chengs to calculate the 90 % upper limit (shown in JUNO-doc-5854):

    chi_squared is given by: chi_2 = 2 * sum(n_i * ln(n_i / v_i) + v_i - n_i)



    to get 90 % limit: 'check up chi_2 belongs to 0.9 and corresponding degree of freedom'

"""
import numpy as np
from matplotlib import pyplot as plt
from gen_spectrum_functions import limit_annihilation_crosssection
from scipy import optimize as op


def chi_squared_s(s, f_s, array_n, array_v):
    """

    :param s: number of signal events
    :param f_s: normalized spectral shape of the signal contribution
    :param array_n: array of the number of observed events per bin
    :param array_v: array of number of expected events per bin
    :return:
    """
    # add a possible signal contribution to the number of expected events:
    array_expected = s * f_s + array_v

    # calculate the chi_squared:
    c_2 = 2 * np.sum(array_n * np.log(array_n / array_expected) + array_expected - array_n)

    return c_2


def chi_squared(param, f_s, f_b1, f_b2, f_b3, f_b4, f_b5, array_n):
    """

    :param param: number of signal events and background events
    :param f_s: normalized spectral shape of the signal contribution
    :param f_b1: normalized spectral shape of background
    :param f_b2: normalized spectral shape of background
    :param f_b3: normalized spectral shape of background
    :param f_b4: normalized spectral shape of background
    :param f_b5: normalized spectral shape of background
    :param array_n: array of number of observed events per bin
    :return:
    """
    # get observed number of events:
    s, b1, b2, b3, b4, b5 = param

    # add a possible signal contribution to the number of expected events:
    array_v = s * f_s + b1 * f_b1 + b2 * f_b2 + b3 * f_b3 + b4 * f_b4 + b5 * f_b5

    # calculate the chi_squared:
    c_2 = 2 * np.sum(array_n * np.log(array_n / array_v) + array_v - array_n)

    return c_2


# load bkg spectra:
input_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"
# DSNB background in events pre bin:
DSNB = input_path + "DSNB_bin500keV_PSD.txt"
DSNB = np.loadtxt(DSNB)
# expected number of DSNB events:
N_DSNB_expected = np.sum(DSNB)
# spectral shape of DSNB per bin:
f_DSNB = DSNB / N_DSNB_expected

# reactor background in events pre bin:
reactor = input_path + "Reactor_NH_power36_bin500keV_PSD.txt"
reactor = np.loadtxt(reactor)
# expected number of reactor events:
N_reactor_expected = np.sum(reactor)
# spectral shape of reactor background per bin:
f_reactor = reactor / N_reactor_expected

# CCatmo background on C12 in events pre bin:
CCatmo_C12 = input_path + "CCatmo_onlyC12_Osc1_bin500keV_PSD.txt"
CCatmo_C12 = np.loadtxt(CCatmo_C12)
# expected number of CCatmo events:
N_CCatmo_C12_expected = np.sum(CCatmo_C12)
# spectral shape of CCatmo background per bin:
f_CCatmo_C12 = CCatmo_C12 / N_CCatmo_C12_expected

# CCatmo background on protons in events pre bin:
CCatmo_p = input_path + "CCatmo_onlyP_Osc1_bin500keV_PSD.txt"
CCatmo_p = np.loadtxt(CCatmo_p)
# expected number of CCatmo events:
N_CCatmo_p_expected = np.sum(CCatmo_p)
# spectral shape of CCatmo background per bin:
f_CCatmo_p = CCatmo_p / N_CCatmo_p_expected

# NC atmo background in events pre bin
input_path_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
                "test_10to20_20to30_30to40_40to100_final/"
NCatmo = input_path_NC + "NCatmo_onlyC12_wPSD99_bin500keV.txt"
NCatmo = np.loadtxt(NCatmo)
# expected number of NC atmo events:
N_NCatmo_expected = np.sum(NCatmo)
# spectral shape of NCatmo background per bin:
f_NCatmo = NCatmo / N_NCatmo_expected

""" calculate the number of observed events per bin (background only): """
# n_array = DSNB + reactor + CCatmo_p + CCatmo_C12 + NCatmo

""" calculate the number of expected background events per bin: """
v_array = DSNB + reactor + CCatmo_p + CCatmo_C12 + NCatmo


# set DM masses in MeV:
# DM_mass = np.arange(15, 105, 5)
DM_mass = [50]

# preallocate array, where the 90 % upper limit of S is stored:
S_90_array = []

# loop over DM masses:
for mass in DM_mass:
    print("----------------------------------------------------------------------------")
    print("\nDM mass = {0:d}".format(mass))

    # signal spectrum in events per bin
    signal = input_path + "signal_DMmass{0:d}_bin500keV_PSD.txt".format(mass)
    signal = np.loadtxt(signal)
    # expected number of signal events:
    N_signal_expected = np.sum(signal)
    # normalize to 1 to get spectral shape:
    f_signal = signal / N_signal_expected

    # minimize chi_2 with respect to the free parameters s, b1, b2, b3, b4, b5:
    # guess of the parameters (as guess, the observed number of events from the simulated spectrum are used)
    # (np.array of float):
    parameter_guess = np.array([N_signal_expected, N_DSNB_expected, N_CCatmo_p_expected, N_CCatmo_C12_expected,
                                N_reactor_expected, N_NCatmo_expected])

    # bounds of the parameters (parameters have to be positive or zero) (tuple):
    bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

    # Minimize chi_squared function with the L-BFGS-B method for the defined bounds above:
    result_chi_squared = op.minimize(chi_squared, parameter_guess, args=(f_signal, f_DSNB, f_CCatmo_p, f_CCatmo_C12,
                                                                         f_reactor, f_NCatmo, v_array),
                                     method='L-BFGS-B', bounds=bnds, options={'disp': None})

    # get the best-fit parameters from the minimization (float):
    N_signal_bf, N_DSNB_bf, N_CCatmo_p_bf, N_CCatmo_C12_bf, N_reactor_bf, N_NCatmo_bf = result_chi_squared["x"]

    # observed number of background events according to best fit values:
    n_array = (N_DSNB_bf * f_DSNB + N_CCatmo_p_bf * f_CCatmo_p + N_CCatmo_C12_bf * f_CCatmo_C12 +
               N_reactor_bf * f_reactor + N_NCatmo_bf * f_NCatmo)

    # array that represents S:
    array_S = np.arange(0, 100, 0.001)

    # array where corresponding chi_2 is stored:
    chi_2_array = []

    for S in array_S:

        chi_2 = chi_squared_s(S, f_signal, n_array, v_array)

        chi_2_array.append(chi_2)

    plt.plot(array_S, chi_2_array)
    plt.grid()
    plt.show()








