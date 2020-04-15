""" script to compare the results of the DM simulations (mean of signal events, mean of S90, 90% limit of
    neutrino flux and 90% limit of annihilation cross-section):
"""
import numpy as np
from matplotlib import pyplot as plt
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_annihilation_crosssection
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_neutrino_flux


def get_results_from_analysis(path_input, masses, n_target, time_sec, j_avg, detection_eff, m_neutron, m_proton,
                              m_positron):
    """

    :param path_input: path, where simulation output is saved
    :param dm_mass: array of Dark Matter masses in MeV
    :param n_target: number of free protons
    :param time_sec: time in seconds
    :param j_avg: angular averaged DM intensity over whole Milky Way
    :param detection_eff: detection efficiency (IBD efficiency * muon exposure ratio)
    :param m_neutron: mass of neutron in MeV
    :param m_proton: mass of proton in MeV
    :param m_positron: mass of positron in MeV
    :return:
    """
    signal = []
    s90 = []
    flux = []
    xsec = []
    reactor = []
    ccatmo = []
    ncatmo = []
    dsnb = []
    fn = []

    for dm_mass in masses:

        # load file result_dataset_output_{}.txt:
        results_simu = np.loadtxt(path_input + "/dataset_output_{0:d}/result_mcmc/result_dataset_output_{0:d}.txt"
                                  .format(dm_mass))

        signal.append(results_simu[4])
        s90.append(results_simu[8])
        # calculate 90 % limit of neutrino flux:
        flux_value = limit_neutrino_flux(results_simu[8], dm_mass, n_target, time_sec, detection_eff, m_neutron,
                                         m_proton, m_positron)
        flux.append(flux_value)

        # calculate 90 % limit of annihilation cross-section:
        xsec_value = limit_annihilation_crosssection(results_simu[8], dm_mass, j_avg, n_target, time_sec, detection_eff,
                                                     m_neutron, m_proton, m_positron)
        xsec.append(xsec_value)

        reactor.append(results_simu[23])
        ccatmo.append(results_simu[18])
        ncatmo.append(results_simu[28])
        dsnb.append(results_simu[13])
        fn.append(results_simu[33])

    return signal, s90, flux, xsec, reactor, ccatmo, ncatmo, dsnb, fn


# path to the results of simulation 1 (flat prior of signal):
path_simu_1 = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN/simulation_1"
# path to the results of simulation 2 (pessimistic prior of signal):
path_simu_2 = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN/simulation_2"
# path to the results of simulation 3 (flat prior of signal, gaussian of background wider):
path_simu_3 = "/home/astro/blum/PhD/work/MeVDM_JUNO/S90_DSNB_CCatmo_reactor_NCatmo_FN/simulation_3"

# define DM masses:
Masses = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
N_target = 1.45 * 10**33
Time_sec = 10 * 3.156 * 10 ** 7
# IBD efficiency * exposure ratio muon veto * (1 - PSD IBD suppression)
Detection_eff = 6.706E-01 * 9.717E-01 * (1 - 0.1108)
M_neutron = 939.56536
M_proton = 938.27203
M_positron = 0.51099892
J_avg = 5.0

# read result_dataset_output_{}.txt file:
(signal_simu_1, s90_simu_1, flux_simu_1, xsec_simu_1, reactor_simu_1, ccatmo_simu_1, ncatmo_simu_1, dsnb_simu_1,
 fn_simu_1) = get_results_from_analysis(path_simu_1, Masses, N_target, Time_sec, J_avg, Detection_eff, M_neutron,
                                        M_proton, M_positron)
(signal_simu_2, s90_simu_2, flux_simu_2, xsec_simu_2, reactor_simu_2, ccatmo_simu_2, ncatmo_simu_2, dsnb_simu_2,
 fn_simu_2) = get_results_from_analysis(path_simu_2, Masses, N_target, Time_sec, J_avg, Detection_eff, M_neutron,
                                        M_proton, M_positron)
(signal_simu_3, s90_simu_3, flux_simu_3, xsec_simu_3, reactor_simu_3, ccatmo_simu_3, ncatmo_simu_3, dsnb_simu_3,
 fn_simu_3) = get_results_from_analysis(path_simu_3, Masses, N_target, Time_sec, J_avg, Detection_eff, M_neutron,
                                        M_proton, M_positron)

# calculate ratio between cross-section of simu_1 and simu_2:
ratio_xsec_12 = np.asarray(xsec_simu_1) / np.asarray(xsec_simu_2)
print("\nratio xsec_1 / xsec_2:")
print(ratio_xsec_12)
print(max(ratio_xsec_12))
ratio_xsec_13 = np.asarray(xsec_simu_1) / np.asarray(xsec_simu_3)
print("\nratio xsec_1 / xsec_3:")
print(ratio_xsec_13)
print(max(ratio_xsec_13))

h1 = plt.figure(1, figsize=(15, 8))
plt.plot(Masses, signal_simu_1, "rx:", label="simu_1: flat prior")
plt.plot(Masses, signal_simu_2, "bx:", label="simu_2: pessimistic prior")
plt.plot(Masses, signal_simu_3, "gx:", label="simu_3: flat prior")
plt.xlabel("DM mass in MeV")
plt.ylabel("number of signal events")
plt.legend()
plt.grid()

h2 = plt.figure(2, figsize=(15, 8))
plt.plot(Masses, s90_simu_1, "rx:", label="simu_1: flat prior")
plt.plot(Masses, s90_simu_2, "bx:", label="simu_2: pessimistic prior")
plt.plot(Masses, s90_simu_3, "gx:", label="simu_3: flat prior")
plt.xlabel("DM mass in MeV")
plt.ylabel("90 % limit of signal events")
plt.legend()
plt.grid()

h3 = plt.figure(3, figsize=(15, 8))
plt.plot(Masses, flux_simu_1, "rx:", label="simu_1: flat prior")
plt.plot(Masses, flux_simu_2, "bx:", label="simu_2: pessimistic prior")
plt.plot(Masses, flux_simu_3, "gx:", label="simu_3: flat prior")
plt.xlabel("DM mass in MeV")
plt.ylabel("90 % limit of neutrino flux")
plt.legend()
plt.grid()

h4 = plt.figure(4, figsize=(15, 8))
plt.plot(Masses, xsec_simu_1, "rx:", label="simu_1: flat prior")
plt.plot(Masses, xsec_simu_2, "bx:", label="simu_2: pessimistic prior")
plt.plot(Masses, xsec_simu_3, "gx:", label="simu_3: flat prior")
plt.xlabel("DM mass in MeV")
plt.ylabel("90 % limit of annihilation cross-section")
plt.legend()
plt.grid()

h5 = plt.figure(5, figsize=(15, 8))
plt.plot(Masses, reactor_simu_1, "rx:", label="simu_1: sigma = mu / 2")
plt.plot(Masses, reactor_simu_2, "bx:", label="simu_2: sigma = mu / 2")
plt.plot(Masses, reactor_simu_3, "gx:", label="simu_3: sigma = mu * 2")
plt.xlabel("DM mass in MeV")
plt.ylabel("number of reactor events")
plt.legend()
plt.grid()

h6 = plt.figure(6, figsize=(15, 8))
plt.plot(Masses, ccatmo_simu_1, "rx:", label="simu_1: sigma = mu / 2")
plt.plot(Masses, ccatmo_simu_2, "bx:", label="simu_2: sigma = mu / 2")
plt.plot(Masses, ccatmo_simu_3, "gx:", label="simu_3: sigma = mu * 2")
plt.xlabel("DM mass in MeV")
plt.ylabel("number of atmo. CC events")
plt.legend()
plt.grid()

h7 = plt.figure(7, figsize=(15, 8))
plt.plot(Masses, ncatmo_simu_1, "rx:", label="simu_1: sigma = mu / 2")
plt.plot(Masses, ncatmo_simu_2, "bx:", label="simu_2: sigma = mu / 2")
plt.plot(Masses, ncatmo_simu_3, "gx:", label="simu_3: sigma = mu * 2")
plt.xlabel("DM mass in MeV")
plt.ylabel("number of atmo. NC events")
plt.legend()
plt.grid()

h8 = plt.figure(8, figsize=(15, 8))
plt.plot(Masses, dsnb_simu_1, "rx:", label="simu_1: sigma = mu / 2")
plt.plot(Masses, dsnb_simu_2, "bx:", label="simu_2: sigma = mu / 2")
plt.plot(Masses, dsnb_simu_3, "gx:", label="simu_3: sigma = mu * 2")
plt.xlabel("DM mass in MeV")
plt.ylabel("number of DSNB events")
plt.legend()
plt.grid()

h9 = plt.figure(9, figsize=(15, 8))
plt.plot(Masses, fn_simu_1, "rx:", label="simu_1: sigma = mu / 2")
plt.plot(Masses, fn_simu_2, "bx:", label="simu_2: sigma = mu / 2")
plt.plot(Masses, fn_simu_3, "gx:", label="simu_3: sigma = mu * 2")
plt.xlabel("DM mass in MeV")
plt.ylabel("number of fast neutron events")
plt.legend()
plt.grid()

plt.show()










