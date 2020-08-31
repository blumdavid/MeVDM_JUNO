""" script to do a hypothesis test like described in paper "signal discovery in sparse spectra: a Bayesian analysis"
    used in GERDA.



"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.special import factorial


def ln_p_h_spectrum(p_sp_h, p_0_h, p_sp_hbar, p_0_hbar):
    """
    ln of the conditional probability for hypothesis H (that the observed spectrum is due to background only)
    to be true or not, given the measured spectrum (equation 2)

    :param p_sp_h: conditional probability to find the observed spectrum given that H is true or not true (equ. 5)
    :param p_0_h: prior probability for H (equ. 18)
    :param p_sp_hbar: conditional probability to find the observed spectrum given that Hbar (hypothesis that the
                            signal process contributes to the spectrum (signal+ bkg)) is true or not true (equ. 6)
    :param p_0_hbar: prior probability for H (equ. 19)

    :return: ln of p(H|spectrum)
    """
    # ln of equation 4:
    ln_p_spectrum = np.log(p_sp_h * p_0_h + p_sp_hbar * p_0_hbar)

    # ln of equation 2:
    ln_p_h_sp = np.log(p_sp_h) + np.log(p_0_h) - ln_p_spectrum

    return ln_p_h_sp


def p_spectrum_h_before_int(b1, b2, b3, b4, b5, array_f_1, array_f_2, array_f_3, array_f_4, array_f_5, array_n,
                            b1_true, b2_true, b3_true, b4_true, b5_true):
    """
    conditional probability to find the observed spectrum given that H is true or not true (equ. 5)

    BEFORE INTEGRATION

    :param b1:
    :param b2:
    :param b3:
    :param b4:
    :param b5:
    :param array_f_1:
    :param array_f_2:
    :param array_f_3:
    :param array_f_4:
    :param array_f_5:
    :param array_n:
    :param b1_true:
    :param b2_true:
    :param b3_true:
    :param b4_true:
    :param b5_true:
    :return:
    """
    # get variables:
    int_function = (p_spectrum_b(b1, b2, b3, b4, b5, array_f_1, array_f_2, array_f_3, array_f_4, array_f_5, array_n)
                    * p_0_b(b1, b1_true) * p_0_b(2, b2_true) * p_0_b(b3, b3_true) * p_0_b(b4, b4_true)
                    * p_0_b(b5, b5_true))

    # set the ranges of integration:
    # ranges = [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]

    # integrate the function int_function over B1, B2, B3, B4, B5 for ranges:
    # p_sp_hbar, absolute_error_2 = integrate.nquad(int_function, ranges)

    return int_function


def p_spectrum_hbar_before_int(s, b1, b2, b3, b4, b5, array_f_s, array_f_1, array_f_2, array_f_3, array_f_4, array_f_5,
                               array_n, s_max, b1_true, b2_true, b3_true, b4_true, b5_true):
    """
    conditional probability to find the observed spectrum given that Hbar is true or not true (equ. 6)

    BEFORE INTEGRATION

    :param s:
    :param b1:
    :param b2:
    :param b3:
    :param b4:
    :param b5:
    :param array_f_s:
    :param array_f_1:
    :param array_f_2:
    :param array_f_3:
    :param array_f_4:
    :param array_f_5:
    :param array_n:
    :param s_max:
    :param b1_true:
    :param b2_true:
    :param b3_true:
    :param b4_true:
    :param b5_true:
    :return:
    """
    int_function = (p_spectrum_s_b(s, b1, b2, b3, b4, b5, array_f_s, array_f_1, array_f_2, array_f_3, array_f_4,
                                   array_f_5, array_n)
                    * p_0_s(s, s_max) * p_0_b(b1, b1_true) * p_0_b(2, b2_true) * p_0_b(b3, b3_true) *
                    p_0_b(b4, b4_true) * p_0_b(b5, b5_true))

    return int_function


def p_0_s(s, s_max):
    """
    prior probability of number of signal events. Flat distribution is assumed (equ. 20)

    :param s: variable (number of signal events)
    :param s_max: maximum value of number of signal events from current limits

    :return:
    """
    if 0.0 <= s <= s_max:
        prior = 1.0 / s_max
    else:
        prior = 0.0

    return prior


def p_0_b(b, b_true):
    """
    prior probabilities of number of background events p_0_b1

    Gaussian around expected number of events (b_true) with width of b_true -> 'poorly know background' of equ. 21.

    Same for all background spectra.

    :param b: variable (number of background events B)
    :param b_true: expected value of B (from simulation)
    :return:
    """
    def exp_func(bkg, mu1, sigma1):
        return np.exp(-(bkg - mu1)**2 / (2 * sigma1**2))

    mu = b_true
    sigma = b_true * 1

    if b >= 0.0:
        # integrate exp_function from 0 to 100:
        integral, abs_err = integrate.quad(exp_func, 0, 100, args=(mu, sigma))

        p_value = exp_func(b, mu, sigma) / integral

    else:
        p_value = 0.0

    return p_value


def p_spectrum_b(b1, b2, b3, b4, b5, array_f_1, array_f_2, array_f_3, array_f_4, array_f_5, array_n):
    """
    conditional probabilities to obtain the measured spectrum (equ. 8), function of B1, B2, B3, B4, B5

    :param b1:
    :param b2:
    :param b3:
    :param b4:
    :param b5:
    :param array_f_1: normalized shape of B1 (array of energy bins)
    :param array_f_2: normalized shape of B2 (array of energy bins)
    :param array_f_3: normalized shape of B3 (array of energy bins)
    :param array_f_4: normalized shape of B4 (array of energy bins)
    :param array_f_5: normalized shape of B5 (array of energy bins)
    :param array_n: observed number of events (array of energy bins)
    :return:
    """
    # calculate array lambda (expected number of events):
    array_lambda = b1 * array_f_1 + b2 * array_f_2 + b3 * array_f_3 + b4 * array_f_4 + b5 * array_f_5

    # calculate p_sp_b:
    p_sp_b = np.prod(array_lambda ** array_n / factorial(array_n, exact=False) * np.exp(-array_lambda))

    return p_sp_b


def p_spectrum_s_b(s, b1, b2, b3, b4, b5, array_f_s, array_f_1, array_f_2, array_f_3, array_f_4, array_f_5, array_n):
    """
    conditional probabilities to obtain the measured spectrum (equ. 9),
    function of S, B1, B2, B3, B4, B5

    :param s:
    :param b1:
    :param b2:
    :param b3:
    :param b4:
    :param b5:
    :param array_f_s: normalized shape of S (array of energy bins)
    :param array_f_1: normalized shape of B1 (array of energy bins)
    :param array_f_2: normalized shape of B2 (array of energy bins)
    :param array_f_3: normalized shape of B3 (array of energy bins)
    :param array_f_4: normalized shape of B4 (array of energy bins)
    :param array_f_5: normalized shape of B5 (array of energy bins)
    :param array_n: observed number of events (array of energy bins)
    :return:
    """
    # calculate array lambda (expected number of events):
    array_lambda = s * array_f_s + b1 * array_f_1 + b2 * array_f_2 + b3 * array_f_3 + b4 * array_f_4 + b5 * array_f_5

    # calculate p_sp_b:
    p_sp_s_b = np.prod(array_lambda ** array_n / factorial(array_n, exact=False) * np.exp(-array_lambda))

    return p_sp_s_b


# set DM mass that should be investigated:
mass_DM = [50.0]
# mass_DM = np.arange(15, 100+5, 5)

# criterion for discovery:
discovery = np.log(0.0001)

# criterion for evidence:
evidence = np.log(0.01)

# prior probability for H:
p_0_H = 0.5
# prior probability for Hbar:
p_0_Hbar = 0.5

# output path:
output_path = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo/test_hypothesis/"

""" Load simulated spectra from file (np.array of float): """
# load bkg spectra:
path_simu = "/home/astro/blum/PhD/work/MeVDM_JUNO/gen_spectrum_v4/S90_DSNB_CCatmo_reactor_NCatmo_FN/"
path_simu_NC = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
               "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
               "test_10to20_20to30_30to40_40to100_final/"
# DSNB spectrum in events/bin:
file_DSNB = path_simu + "/DSNB_bin500keV_PSD.txt"
spectrum_DSNB = np.loadtxt(file_DSNB)
# reactor spectrum in events/bin:
file_reactor = path_simu + "/Reactor_NH_power36_bin500keV_PSD.txt"
spectrum_Reactor = np.loadtxt(file_reactor)
# atmo. CC background on proton in events/bin:
file_CCatmo_p = path_simu + "/CCatmo_onlyP_Osc1_bin500keV_PSD.txt"
spectrum_CCatmo_p = np.loadtxt(file_CCatmo_p)
# atmo. CC background on C12 in events/bin:
file_CCatmo_C12 = path_simu + "/CCatmo_onlyC12_Osc1_bin500keV_PSD.txt"
spectrum_CCatmo_C12 = np.loadtxt(file_CCatmo_C12)
# atmo. NC background in events/bin:
file_NCatmo = path_simu_NC + "/NCatmo_onlyC12_wPSD99_bin500keV.txt"
spectrum_NCatmo = np.loadtxt(file_NCatmo)

""" Define the energy window, where spectrum of virtual experiment and simulated spectrum is analyzed
    (from min_E_cut in MeV to max_E_cut in MeV): """
min_E_visible = 10.0
max_E_visible = 100.0
interval_E_visible = 0.5

""" 'true' values of the parameters: """
# expected number of DSNB background events in the energy window (float):
B_DSNB_true = np.sum(spectrum_DSNB)
# expected number of CCatmo background events on protons in the energy window (float):
B_CCatmo_p_true = np.sum(spectrum_CCatmo_p)
# expected number of CCatmo background events on C12 in the energy window (float):
B_CCatmo_C12_true = np.sum(spectrum_CCatmo_C12)
# expected number of Reactor background events in the energy window (float):
B_Reactor_true = np.sum(spectrum_Reactor)
# expected number of NCatmo background events in the energy window (float):
B_NCatmo_true = np.sum(spectrum_NCatmo)

""" fractions (normalized shapes) of signal and background spectra: """
# Fraction of DSNB background (np.array of float):
fraction_DSNB = spectrum_DSNB / B_DSNB_true
# Fraction of CCatmo background on protons (np.array of float):
fraction_CCatmo_p = spectrum_CCatmo_p/ B_CCatmo_p_true
# Fraction of CCatmo background on C12 (np.array of float):
fraction_CCatmo_C12 = spectrum_CCatmo_C12 / B_CCatmo_C12_true
# Fraction of reactor background (np.array of float):
fraction_Reactor = spectrum_Reactor / B_Reactor_true
# Fraction of NCatmo background (np.array of float):
fraction_NCatmo = spectrum_NCatmo / B_NCatmo_true

# loop over DM masses:
for mass in mass_DM:
    print("----------------------------------------------------------------------------")
    print("\nDM mass = {0:.0f}\n".format(mass))

    # signal spectrum in events/bin:
    file_signal = path_simu + "/signal_DMmass{0:.0f}_bin500keV_PSD.txt".format(mass)
    spectrum_Signal = np.loadtxt(file_signal)
    file_info_signal = path_simu + "/signal_info_DMmass{0:.0f}_bin500keV_PSD.txt".format(mass)
    info_signal = np.loadtxt(file_info_signal)

    # expected number of signal events in the energy window (float):
    S_true = np.sum(spectrum_Signal)

    # Fraction of DM signal (np.array of float):
    fraction_Signal = spectrum_Signal / S_true

    # maximum value of signal events consistent with existing limits (assuming the 'new' 90 % upper limit for
    # annihilation cross-section of Super-K from paper "Dark matter-neutrino interactions through the lens of their
    # cosmological implications" (PhysRevD.97.075039 of Olivares-Del Campo), for the description and calculation see
    # limit_from_SuperK.py)
    # INFO-me: S_max is assumed from the limit on the annihilation cross-section of Super-K (see limit_from_SuperK.py)
    S_max = 60

    """ define the array of observed number of events ('the measured data'): """
    Data = (spectrum_Signal + spectrum_DSNB + spectrum_Reactor + spectrum_CCatmo_p + spectrum_CCatmo_C12 +
            spectrum_NCatmo)

    """ integrate p_spectrum_b * p_0_bkgs: """
    # set the ranges of integration:
    ranges_b = [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]

    # set opts of integration:
    opts_b = [{}, {}, {}, {}, {}]

    # integrate the function p_spectrum_h_before_int() over B1, B2, B3, B4, B5 for ranges:
    p_spectrum_H, absolute_error_1 = integrate.nquad(p_spectrum_h_before_int, ranges=ranges_b,
                                                     args=(fraction_DSNB, fraction_Reactor, fraction_CCatmo_p,
                                                           fraction_CCatmo_C12, fraction_NCatmo, Data, B_DSNB_true,
                                                           B_Reactor_true, B_CCatmo_p_true, B_CCatmo_C12_true,
                                                           B_NCatmo_true),
                                                     opts=opts_b, full_output=False)

    """ integrate p_spectrum_s_b * p_0_s * p_0_bkgs: """
    # set the ranges of integration:
    ranges_s_b = [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]

    # set opts of integration:
    opts_s_b = [{}, {}, {}, {}, {}, {}]

    # integrate the function p_spectrum_h_before_int() over B1, B2, B3, B4, B5 for ranges:
    p_spectrum_Hbar, absolute_error_2 = integrate.nquad(p_spectrum_hbar_before_int, ranges=ranges_s_b,
                                                        args=(fraction_Signal, fraction_DSNB, fraction_Reactor,
                                                              fraction_CCatmo_p, fraction_CCatmo_C12, fraction_NCatmo,
                                                              Data, S_max, B_DSNB_true,
                                                              B_Reactor_true, B_CCatmo_p_true, B_CCatmo_C12_true,
                                                              B_NCatmo_true),
                                                        opts=opts_s_b, full_output=False)

    """ calculate the ln of the conditional probability ofr H to be true of not, given the measured spectrum: """
    # ln of conditional probability ofr H to be true of not, given the measured spectrum:
    ln_p_H_spectrum = ln_p_h_spectrum(p_spectrum_H, p_0_H, p_spectrum_Hbar, p_0_Hbar)

    # check criterion for discovery and evidence:
    if ln_p_H_spectrum <= discovery:
        print("DISCOVERY\n")
    elif discovery < ln_p_H_spectrum <= evidence:
        print("EVIDENCE\n")
    else:
        print("no discovery, no evidence (ln[ p(H|spectrum) ]) = {0}".format(ln_p_H_spectrum))







