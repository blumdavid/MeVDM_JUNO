""" Script to calculate the limit on the number of signal events from the limit on the annihilation cross-section
    from Super-Kamiokande in the paper 0710.5420.pdf

    The result can be used to define the maximum of the prior probability of the signal contribution (S_max) in
    the script analyze_spectra_v4_*.py

"""
import numpy as np
from MeVDM_JUNO.source.gen_spectrum_functions import sigma_ibd

# define the DM masses in MeV:
DM_mass = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

# mass of positron in MeV (reference PDG 2016) (float constant):
MASS_POSITRON = 0.51099892
# mass of proton in MeV (reference PDG 2016) (float constant):
MASS_PROTON = 938.27203
# mass of neutron in MeV (reference PDG 2016) (float constant):
MASS_NEUTRON = 939.56536
# difference MASS_NEUTRON - MASS_PROTON in MeV (float):
DELTA = MASS_NEUTRON - MASS_PROTON

# total time-exposure in seconds, 10 years (float):
time = 10 * 3.156 * 10 ** 7
# Number of free protons (target particles) for IBD in JUNO (page 18 of JUNO DesignReport) (float):
N_target = 1.45 * 10 ** 33
# detection efficiency of IBD in JUNO, from physics_report.pdf, page 40, table 2.1
# (combined efficiency of energy cut, time cut, vertex cut, Muon veto, fiducial volume (only r<17m)) (float):
detection_eff = 0.73

# solar radius circle in cm, 8.5 kiloparsec, 1kpc = 3.086*10**21 cm (float):
r_solar = 8.5 * 3.086 * 10 ** 21
# normalizing DM density, in MeV/cm**3 (float):
rho_0 = 0.3 * 1000
# angular-averaged dark matter intensity over the whole Milky Way (float):
J_avg_5 = 5
J_avg_1 = 1.3

# Limit on the self-annihilation cross-section from Super-K in cm**3/s (0710.5420.pdf) (as a function of DM_mass
# from above) (the limits are taken from the digitized Plot of 0710.5420 for J_avg=5)
# (data is saved in limit_SuperK_digitized.csv)) (np.array of float):
sigma_annihilation_Javg5 = np.array([6.43908E-26, 4.709E-26, 6.54902E-26, 1.18448E-25, 1.83953E-25, 2.33156E-25,
                                     2.25393E-25, 1.63394E-25, 1.03444E-25, 7.49894E-26, 7.24927E-26, 8.44249E-26,
                                     9.66705E-26, 1.42696E-25, 2.29242E-25, 4.58949E-25, 8.00000E-25])

# Limit on the self-annihilation cross-section from Super-K in cm**3/s (0710.5420.pdf) (as a function of DM_mass
# from above) (the limits are taken from the digitized Plot of 0710.5420 for J_avg=1.3)
# (data is saved in limit_SuperK_digitized.csv)) (np.array of float):
sigma_annihilation_Javg1 = np.array([2.5E-25, 1.8E-25, 2.8E-25, 4.0E-25, 6.9E-25, 8.9E-25, 8.8E-25, 6.3E-25, 3.8E-25,
                                     3.0E-25, 2.8E-25, 3.0E-25, 3.8E-25, 5.0E-25, 7.9E-25, 1.5E-24, 3.0E-24])

# Cross-section of the Inverse Beta Decay in cm**2 (np.array of float):
sigma_IBD = sigma_ibd(DM_mass, DELTA, MASS_POSITRON)

# Electron-Antineutrino flux from DM annihilation in 1/(s*cm**2) (np.array of float):
# for sigma_annihilation with J_avg = 5:
flux_nuEbar_Javg5 = sigma_annihilation_Javg5 * J_avg_5 * r_solar * rho_0**2 / (2 * DM_mass**2 * 3)

# Electron-Antineutrino flux from DM annihilation in 1/(s*cm**2) (np.array of float):
# for sigma_annihilation with J_avg = 1.3:
flux_nuEbar_Javg1 = sigma_annihilation_Javg1 * J_avg_1 * r_solar * rho_0**2 / (2 * DM_mass**2 * 3)

# Number of signal events assuming the limit on the annihilation cross-section of Super-K from 0710.5420.pdf
# for sigma_annihilation with J_avg = 5:
# (np.array of float):
number_signal_Javg5 = sigma_IBD * flux_nuEbar_Javg5 * N_target * time * detection_eff

# Number of signal events assuming the limit on the annihilation cross-section of Super-K from 0710.5420.pdf
# for sigma_annihilation with J_avg = 1.3:
# (np.array of float):
number_signal_Javg1 = sigma_IBD * flux_nuEbar_Javg1 * N_target * time * detection_eff

print("DM masses in MEV:")
print(DM_mass)
print("Number of signal events assuming the limit on the annihilation cross-section of Super-K for DM masses of above "
      "(J_avg = 5):")
print(number_signal_Javg5)
print("Number of signal events assuming the limit on the annihilation cross-section of Super-K for DM masses of above "
      "(J_avg = 1.3):")
print(number_signal_Javg1)
