""" Script to test the behaviour of the convolution of the theoretical spectrum with a gaussian distribution to
    calculate the visible spectrum in the detector:
    """

import numpy as np
from matplotlib import pyplot
import work.MeVDM_JUNO.source.gen_spectrum_functions as f1

# mass of positron in MeV (float constant):
mass_positron = 0.51099892
# mass of proton in MeV (float constant):
mass_proton = 938.27203
# mass of neutron in MeV (float constant):
mass_neutron = 939.56536

""" values that are given: e_neutrino, s_theo, e_visible(e_neutrino), sigma(e_visible(e_neutrino)): """
# neutrino energy in MeV:
binning_e_neutrino = 1
e_neutrino = np.arange(11, 20+binning_e_neutrino, binning_e_neutrino)
print("e_neutrino = {0}".format(e_neutrino))
# theoretical spectrum as function of e_neutrino in 1/MeV:
# s_theo = np.array([0, 0, 1, 3, 4, 7, 8, 7, 5, 5])
s_theo = np.array([0, 0, 0, 0, 0, 5, 0, 0, 0, 0])
# s_theo = np.random.randint(0, 10, len(e_neutrino))
# s_theo = np.arange(0, len(e_neutrino))
# s_theo = np.zeros(len(e_neutrino))
# s_theo[400] = 50
print("s_theo = {0}".format(s_theo))
# visible energy as function of the neutrino energy in the detector (sichtbare Energie) in MeV
# (is a function of the positron energy, positron energy is a function of the neutrino energy):
e_visible = f1.correlation_vis_neutrino(e_neutrino, mass_proton, mass_neutron, mass_positron)
print("e_visible = {0}".format(e_visible))
# energy resolution of the detector is describe by sigma, which is a function of the neutrino energy
# (is a function of the visible energy, visible energy is function of the positron energy, positron energy is a function
# of the neutrino energy):
sigma = f1.energy_resolution(e_visible)
print("sigma = {0}".format(sigma))

""" values for the measured (detected) energy: """
# measured or detected energy in the detector in MeV:
binning_e_measured = 2
e_measured = np.arange(10, 18+binning_e_measured, binning_e_measured)
print("e_measured = {0}".format(e_measured))


""" test 1 ("Berechnung 1" in my notes): 
    1. calculate for every entry in e_neutrino the values of s_theo, e_visible and sigma. With these values 
    calculate for every entry in e_measured value for the measured spectrum
    2. then you have the measured spectrum as function of e_measured for every entry in e_neutrino
    3. add these measured spectra elementwise to get the whole measured spectrum
    """
s_measured_test1 = np.zeros(len(e_measured))

# loop over entries in e_neutrino ('for every entry in e_neutrino'):
for index1 in np.arange(len(e_neutrino)):
    # preallocate array, equal to s_measured_test1 for e_neutrino[index1]:
    s_measured1 = np.array([])
    # for 1 entry in e_neutrino, loop over entries in e_measured ('for every entry in e_measured'):
    for index2 in np.arange(len(e_measured)):
        # spectrum measured of e_neutrino[index1] for energy e_measured[index2]:
        s_measured_ForOneEntryInEneutrino = (s_theo[index1] * 1/(np.sqrt(2*np.pi)*sigma[index1]) *
                                             np.exp(-0.5 * (e_measured[index2] - e_visible[index1])**2 /
                                                    sigma[index1]**2))
        # print(s_measured_ForOneEntryInEneutrino)
        # append value for e_measured[index2] to s_measured1 to get an array:
        s_measured1 = np.append(s_measured1, s_measured_ForOneEntryInEneutrino)

    # print("measured spectrum for fix e_neutrino as function of e_measured = {0}".format(s_measured1))
    # add up the measured spectra elementwise:
    s_measured_test1 = s_measured_test1 + s_measured1
    # print("total measured spectrum = {0}".format(s_measured_test1))

# print("total measured spectrum with test 1 = {0}".format(s_measured_test1))

""" test 2 and 3 ("Berechnung 2, Test 2 und Test 3" in my notes):
    1. go through the entries of e_measured
    2. for every entry of e_measured, go through the entries of e_neutrino
    3. calculate the values of s_theo, e_visible, and sigma for this value of e_neutrino
    4. then with s_theo, e_visible, sigma, e_neutrino and e_measured calculate the value of s_measured
    5. then do the same for the second entry in e_neutrino (and for all values in e_neutrino)
    6. then go to the second value of e_measured and do the point 2 to 5 again
    7. for every entry in e_measured, you get the measured spectrum s_measured as function of e_neutrino
    test 2: sum entries of s_measured over e_neutrino to get the total spectrum
    test 3: integrate the entries of s_measured over e_neutrino with np.trapz to get the total spectrum
    """
s_measured_test2 = np.array([])
s_measured_test3 = np.array([])

# loop over entries in e_measured ('for every entry in e_measured'):
for index2 in np.arange(len(e_measured)):
    # preallocate array, equal to s_measured_test2 for e_measured[index2]:
    s_measured2 = np.array([])
    # for 1 entry in e_measured, loop over entries in e_neutrino ('for every entry in e_neutrino'):
    for index1 in np.arange(len(e_neutrino)):
        # spectrum measured of e_measured[index2] for the energy e_neutrino[index1]:
        s_measured_ForOneEntryInEmeasured = (s_theo[index1] * 1/(np.sqrt(2*np.pi)*sigma[index1]) *
                                             np.exp(-0.5 * (e_measured[index2] - e_visible[index1])**2 /
                                                    sigma[index1]**2))
        # append value for e_neutrino[index1] to s_measured2 to get an array:
        s_measured2 = np.append(s_measured2, s_measured_ForOneEntryInEmeasured)

    # print("s_measured for fix e_measured as function of e_neutrino = {0}".format(s_measured2))

    """ test 2: """
    # add up the values of s_measured2 (equal to S_m_Emeasured1MeV(e_neutrino)) :
    s_measured_Emeasured2 = np.sum(s_measured2)
    # print("s_measured for fix e_measured (summed over e_neutrino) = {0}".format(s_measured_Emeasured2))
    # append s_measured_Emeasured2 to the array s_measured_test2:
    s_measured_test2 = np.append(s_measured_test2, s_measured_Emeasured2)
    # print("total measured spectrum = {0}".format(s_measured_test2))

    """ test 3: """
    # integrate s_measured2 over e_neutrino:
    s_measured_Emeasured3 = np.array([np.trapz(s_measured2, e_neutrino)])
    # print("s_measured for fix e_measured (integrated over e_neutrino) = {0}".format(s_measured_Emeasured3))
    # append s_measured_Emeasured3 to the array s_measured_test3:
    s_measured_test3 = np.append(s_measured_test3, s_measured_Emeasured3)
    # print("total measured spectrum with test3 = {0}".format(s_measured_test3))

# print("total measured spectrum with test2 = {0}".format(s_measured_test2))
# print("total measured spectrum with test3 = {0}".format(s_measured_test3))

""" test 4:
    Calculation like in gen_spectrum_v1.py
    Spectrum (theoretical spectrum is convolved with gaussian distribution): 
    """
# Preallocate the spectrum array (np.array):
s_measured_test4 = np.array([])
# convolve the 'theoretical'-spectrum with the gaussian distribution:
for index in np.arange(len(e_measured)):
    # gaussian distribution characterized by e_visible and sigma (np.array of float):
    gauss = (1 / (np.sqrt(2 * np.pi) * sigma) *
             np.exp(-0.5 * (e_measured[index] - e_visible) ** 2 / sigma**2))
    # defines the integrand, which will be integrated over the neutrino energy energy_neutrino (np.array of float):
    integrand = s_theo * gauss
    # integrate the integrand over the neutrino energy e_neutrino to get the value of the spectrum
    # for one visible energy energy_visible[index] in 1/MeV:
    s_measured_Emeasured4 = np.array([np.trapz(integrand, e_neutrino)])
    # append the single values spectrum_signal_index to the array spectrum_signal:
    s_measured_test4 = np.append(s_measured_test4, s_measured_Emeasured4)

print("total measured spectrum with test 4 = {0}".format(s_measured_test4))

""" New Calculation ("Berechnung 3" in my notes):
    Is implemented in function convolution() of script gen_spectrum_functions.py and used 
    in gen_spectrum_v2.py.
    Calculation is similar to test 3, BUT: Integrate the values for 1 entry in e_measured and 1 entry in e_neutrino 
    over the whole bin-width of e_measured[1] to e_measured[2] and at the end divide by the bin-width of e_measured 
    to get the spectrum in units of 1/MeV.
    """
# Preallocate the spectrum array:
s_measured_NEW = np.array([])

# loop over entries in e_measured ('for every entry in e_measured'):
for index2 in np.arange(len(e_measured)):
    # preallocate array (in the notes defined as S_m^(E_m=1) as function of e_neutrino:
    s_m_Em1 = np.array([])
    # for 1 entry in e_measured, loop over entries in e_neutrino ('for every entry in e_neutrino'):
    for index1 in np.arange(len(e_neutrino)):
        # define energy E in MeV, E is in the range of e_measured[index2] and e_measured[index2+1]:
        E = np.arange(e_measured[index2]-binning_e_measured/2, e_measured[index2]+binning_e_measured/2,
                      binning_e_measured/100)
        # s_m of E for the energy e_neutrino[index1], unit=1/MeV**2:
        s_measured_E = (s_theo[index1] * 1/(np.sqrt(2*np.pi)*sigma[index1]) *
                        np.exp(-0.5 * (E - e_visible[index1])**2 / sigma[index1]**2))
        # integrate s_measured_E over E from e_measured[index2] to e_measured[index2+1] ('integrate over the bin'),
        # unit=1/MeV:
        s_measured_ForOneEntryInEmeasured = np.array([np.trapz(s_measured_E, E)])
        print("s_m for one entry of e_m and one entry of e_neutrino (integral over E) = {0}".
              format(s_measured_ForOneEntryInEmeasured))
        # append value for e_neutrino[index1] to s_measuredNEW to get an array, unit=1/MeV:
        s_m_Em1 = np.append(s_m_Em1, s_measured_ForOneEntryInEmeasured)

    print("s_m for one entry in E_m as function of e_neutrino = {0}".format(s_m_Em1))
    # integrate s_m_Em1 over e_neutrino, unit='number of neutrinos':
    s_measuredNEW = np.array([np.trapz(s_m_Em1, e_neutrino)])
    # to consider the binwidth of e_measured, divide the value of s_measuredNEW by the binning of e_measured,
    # unit=1/MeV:
    s_measuredNEW = s_measuredNEW / binning_e_measured
    # append s_measuredNEW to the array s_measured_NEW, unit=1/MeV:
    s_measured_NEW = np.append(s_measured_NEW, s_measuredNEW)

print("total spectrum with the NEW calculation = {0}".format(s_measured_NEW))

h1 = pyplot.figure(1)
pyplot.step(e_neutrino, s_theo, where='mid', label='theo. spectrum as function of e_neutrino')
# pyplot.step(e_measured, s_measured_test1, where='mid', label='visible spectrum with test 1')
# pyplot.step(e_measured, s_measured_test3, where='mid', label='visible spectrum with test 4 (gen_spectrum_v1.py')
pyplot.step(e_measured, s_measured_NEW, where='mid', label='visible spectrum with NEW calculation (gen_spectrum_v2.py)')
pyplot.ylim(ymin=0)
pyplot.xlabel("energy in MeV")
pyplot.ylabel("spectrum in 1/MeV")
pyplot.title("Convolution of theo. spectrum for different calculations (see test_convolution.py")
pyplot.grid()
pyplot.legend()
pyplot.show()
