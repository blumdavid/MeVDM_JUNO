""" script to display the atmospheric neutrino flux of Honda and from FLUKA simulations in the whole energy range from
    10 MeV to 10 TeV for all neutrino flavour (electron, electron-anti, muon, muon-anti):
"""
import numpy as np
from matplotlib import pyplot as plt

# define the neutrino energy in MeV:
energy_interesting = np.arange(13, 100.01, 0.01)
energy_total = np.arange(100, 10000, 1)
energy_norm = np.arange(100, 944.1, 0.1)

""" read data file HONDA_juno-ally-01-01-solmin.d (angular average flux at JUNO site from Honda from 100 MeV to 10 TeV):
"""
# open file:
honda_file = "/home/astro/blum/PhD/paper/atmospheric_neutrinos/HONDA_juno-ally-01-01-solmin.txt"
# read file:
honda_data = np.loadtxt(honda_file)

# neutrino energy in GeV:
energy_honda = []
# neutrino fluxes in 1/(m^2 sec sr GeV):
nu_e_honda = []
nu_e_bar_honda = []
nu_mu_honda = []
nu_mu_bar_honda = []

for index in range(len(honda_data)):
    # neutrino energy in GeV:
    energy_honda.append(honda_data[index][0])
    # nu_e flux:
    nu_e_honda.append(honda_data[index][3])
    # nu_e_bar flux:
    nu_e_bar_honda.append(honda_data[index][4])
    # nu_mu flux:
    nu_mu_honda.append(honda_data[index][1])
    # nu_mu_bar flux:
    nu_mu_bar_honda.append(honda_data[index][2])

# energy in MeV:
energy_honda = 1000 * np.asarray(energy_honda)
# fluxes in 1/(cm^2 * sec * MeV):
nu_e_honda = 10**(-7) * 4*np.pi * np.asarray(nu_e_honda)
nu_e_bar_honda = 10**(-7) * 4*np.pi * np.asarray(nu_e_bar_honda)
nu_mu_honda = 10**(-7) * 4*np.pi * np.asarray(nu_mu_honda)
nu_mu_bar_honda = 10**(-7) * 4*np.pi * np.asarray(nu_mu_bar_honda)

""" get data from FLUKA simulations """
# Neutrino energy in MeV from table 3 from paper 1-s2.0-S0927650505000526-main (np.array of float):
energy_FLUKA = np.array([13, 15, 17, 19, 21, 24, 27, 30, 33, 38, 42, 47, 53, 60, 67, 75, 84, 94, 106, 119, 133, 150,
                         168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 531, 596, 668, 750, 841, 944])

# differential flux in energy for no oscillation for electron-neutrinos for solar average at the site
# of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuE_FLUKA = 10 ** (-7) * np.array([0.696E+05, 0.746E+05, 0.797E+05, 0.874E+05, 0.942E+05, 0.101E+06, 0.103E+06,
                                        0.109E+06, 0.108E+06, 0.107E+06, 0.101E+06, 0.885E+05, 0.696E+05, 0.644E+05,
                                        0.593E+05, 0.543E+05, 0.497E+05, 0.451E+05, 0.406E+05, 0.358E+05, 0.317E+05,
                                        0.273E+05, 0.239E+05, 0.204E+05, 0.170E+05, 0.145E+05, 0.120E+05, 0.996E+04,
                                        0.811E+04, 0.662E+04, 0.527E+04, 0.423E+04, 0.337E+04, 0.266E+04, 0.209E+04,
                                        0.162E+04, 0.124E+04, 0.950E+03])

# differential flux in energy for no oscillation for electron-antineutrinos for solar average at the site
# of Super-Kamiokande, in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuEbar_FLUKA = 10 ** (-4) * np.array([63.7, 69.7, 79.5, 84.2, 89.4, 95.0, 99.3, 103., 104., 101., 96.1,
                                           83.5, 65.9, 60.0, 56.4, 51.4, 46.3, 43.0, 37.2, 32.9, 28.8, 24.9, 21.3,
                                           18.3, 15.4, 12.9, 10.6, 8.80, 7.13, 5.75, 4.60, 3.68, 2.88, 2.28,
                                           1.87, 1.37, 1.06, 0.800])

# differential flux in energy for no oscillation for muon-neutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuMu_FLUKA = 10 ** (-7) * np.array([0.114E+06, 0.124E+06, 0.138E+06, 0.146E+06, 0.155E+06, 0.159E+06, 0.164E+06,
                                         0.181E+06, 0.174E+06, 0.179E+06, 0.178E+06, 0.176E+06, 0.153E+06, 0.131E+06,
                                         0.123E+06, 0.114E+06, 0.107E+06, 0.963E+05, 0.842E+05, 0.727E+05, 0.635E+05,
                                         0.552E+05, 0.477E+05, 0.412E+05, 0.344E+05, 0.284E+05, 0.236E+05, 0.196E+05,
                                         0.158E+05, 0.128E+05, 0.103E+05, 0.820E+04, 0.649E+04, 0.515E+04, 0.398E+04,
                                         0.313E+04, 0.241E+04, 0.182E+04])

# differential flux in energy for no oscillation for muon-antineutrinos for solar average at the site of Super-K,
# in (MeV**(-1) * cm**(-2) * s**(-1)) (np.array of float):
flux_nuMubar_FLUKA = 10 ** (-4) * np.array([116., 128., 136., 150., 158., 162., 170., 196., 177., 182., 183.,
                                            181., 155., 132., 123., 112., 101., 92.1, 82.2, 72.5, 64.0, 55.6, 47.6,
                                            40.8, 34.1, 28.6, 23.5, 19.3, 15.7, 12.6, 10.2, 8.15, 6.48, 5.02, 3.94,
                                            3.03, 2.33, 1.79])

""" normalized FLUKA flux to Honda flux: """
""" For electron neutrinos: """
# interpolate Honda flux from 10 MeV to 10 GeV = 10000 TeV:
flux_nuE_honda_Juno = np.interp(energy_total, energy_honda, nu_e_honda)
# Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuE_FLUKA_interpolated = np.interp(energy_norm, energy_FLUKA, flux_nuE_FLUKA)
# Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuE_HONDA_interpolated = np.interp(energy_norm, energy_honda, nu_e_honda)
# Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
integral_nuE_FLUKA = np.trapz(flux_nuE_FLUKA_interpolated, energy_norm)
# Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
integral_nuE_HONDA = np.trapz(flux_nuE_HONDA_interpolated, energy_norm)
# Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
# (np.array of float):
flux_nuE_FLUKA_interesting = np.interp(energy_interesting, energy_FLUKA, flux_nuE_FLUKA)
# Normalize flux_nuEbar_FLUKA_interesting at Super-K to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2)
# (np.array of float):
flux_nuE_JUNO = flux_nuE_FLUKA_interesting * integral_nuE_HONDA / integral_nuE_FLUKA
norm_factor_nuE = integral_nuE_HONDA / integral_nuE_FLUKA
print("normalization factor nu_e (honda/fluka) = {0:.5f}".format(norm_factor_nuE))
# Normalize the whole FLUKA flux to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2), (contains the
# interesting part from 13 MeV to 115 MeV and the part from 113 MeV to 944 MeV to compare it to HONDA):
flux_nuE_FLUKA_normalized = flux_nuE_FLUKA * integral_nuE_HONDA / integral_nuE_FLUKA

""" For electron antineutrinos: """
# interpolate Honda flux from 10 MeV to 10 GeV = 10000 TeV:
flux_nuEbar_honda_Juno = np.interp(energy_total, energy_honda, nu_e_bar_honda)
# Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuEbar_FLUKA_interpolated = np.interp(energy_norm, energy_FLUKA, flux_nuEbar_FLUKA)
# Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuEbar_HONDA_interpolated = np.interp(energy_norm, energy_honda, nu_e_bar_honda)
# Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
integral_nuEbar_FLUKA = np.trapz(flux_nuEbar_FLUKA_interpolated, energy_norm)
# Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
integral_nuEbar_HONDA = np.trapz(flux_nuEbar_HONDA_interpolated, energy_norm)
# Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
# (np.array of float):
flux_nuEbar_FLUKA_interesting = np.interp(energy_interesting, energy_FLUKA, flux_nuEbar_FLUKA)
# Normalize flux_nuEbar_FLUKA_interesting at Super-K to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2)
# (np.array of float):
flux_nuEbar_JUNO = flux_nuEbar_FLUKA_interesting * integral_nuEbar_HONDA / integral_nuEbar_FLUKA
norm_factor_nuEbar = integral_nuEbar_HONDA / integral_nuEbar_FLUKA
print("normalization factor nu_e_bar (honda/fluka) = {0:.5f}".format(norm_factor_nuEbar))
# Normalize the whole FLUKA flux to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2), (contains the
# interesting part from 13 MeV to 115 MeV and the part from 113 MeV to 944 MeV to compare it to HONDA):
flux_nuEbar_FLUKA_normalized = flux_nuEbar_FLUKA * integral_nuEbar_HONDA / integral_nuEbar_FLUKA

""" For muon neutrinos: """
# interpolate Honda flux from 10 MeV to 10 GeV = 10000 TeV:
flux_nuMu_honda_Juno = np.interp(energy_total, energy_honda, nu_mu_honda)
# Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuMu_FLUKA_interpolated = np.interp(energy_norm, energy_FLUKA, flux_nuMu_FLUKA)
# Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuMu_HONDA_interpolated = np.interp(energy_norm, energy_honda, nu_mu_honda)
# Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
integral_nuMu_FLUKA = np.trapz(flux_nuMu_FLUKA_interpolated, energy_norm)
# Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
integral_nuMu_HONDA = np.trapz(flux_nuMu_HONDA_interpolated, energy_norm)
# Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
# (np.array of float):
flux_nuMu_FLUKA_interesting = np.interp(energy_interesting, energy_FLUKA, flux_nuMu_FLUKA)
# Normalize flux_nuEbar_FLUKA_interesting at Super-K to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2)
# (np.array of float):
flux_nuMu_JUNO = flux_nuMu_FLUKA_interesting * integral_nuMu_HONDA / integral_nuMu_FLUKA
norm_factor_nuMu = integral_nuMu_HONDA / integral_nuMu_FLUKA
print("normalization factor nu_mu (honda/fluka) = {0:.5f}".format(norm_factor_nuMu))
# Normalize the whole FLUKA flux to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2), (contains the
# interesting part from 13 MeV to 115 MeV and the part from 113 MeV to 944 MeV to compare it to HONDA):
flux_nuMu_FLUKA_normalized = flux_nuMu_FLUKA * integral_nuMu_HONDA / integral_nuMu_FLUKA

""" For muon antineutrinos: """
# interpolate Honda flux from 10 MeV to 10 GeV = 10000 TeV:
flux_nuMubar_honda_Juno = np.interp(energy_total, energy_honda, nu_mu_bar_honda)
# Interpolate the flux of FLUKA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuMubar_FLUKA_interpolated = np.interp(energy_norm, energy_FLUKA, flux_nuMubar_FLUKA)
# Interpolate the flux of HONDA to get the differential flux in the energy range from 100 MeV to 950 MeV,
# in 1/(MeV * cm**2 * s) (np.array of float):
flux_nuMubar_HONDA_interpolated = np.interp(energy_norm, energy_honda, nu_mu_bar_honda)
# Calculate the integral of the FLUKA flux in the energy range given by energy_norm (float):
integral_nuMubar_FLUKA = np.trapz(flux_nuMubar_FLUKA_interpolated, energy_norm)
# Calculate the integral of the HONDA flux in the energy range given by energy_norm (float):
integral_nuMubar_HONDA = np.trapz(flux_nuMubar_HONDA_interpolated, energy_norm)
# Interpolate the INTERESTING part of the FLUKA flux in the energy range from 10 MeV to 100 MeV, in 1/(MeV*s*cm**2)
# (np.array of float):
flux_nuMubar_FLUKA_interesting = np.interp(energy_interesting, energy_FLUKA, flux_nuMubar_FLUKA)
# Normalize flux_nuEbar_FLUKA_interesting at Super-K to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2)
# (np.array of float):
flux_nuMubar_JUNO = flux_nuMubar_FLUKA_interesting * integral_nuMubar_HONDA / integral_nuMubar_FLUKA
norm_factor_nuMubar = integral_nuMubar_HONDA / integral_nuMubar_FLUKA
print("normalization factor nu_mu_bar (honda/fluka) = {0:.5f}".format(norm_factor_nuMubar))
# Normalize the whole FLUKA flux to the electron-antineutrino flux at JUNO, in 1/(MeV * s * cm**2), (contains the
# interesting part from 13 MeV to 115 MeV and the part from 113 MeV to 944 MeV to compare it to HONDA):
flux_nuMubar_FLUKA_normalized = flux_nuMubar_FLUKA * integral_nuMubar_HONDA / integral_nuMubar_FLUKA


plt.figure(1, figsize=(10, 5.5))
plt.loglog(energy_total, flux_nuE_honda_Juno, "r--", label="$\\nu_e$-flux of Honda")
plt.loglog(energy_total, flux_nuEbar_honda_Juno, "k--", label="$\\bar{\\nu}_e$-flux of Honda")
plt.loglog(energy_total, flux_nuMu_honda_Juno, "b--", label="$\\nu_\\mu$-flux of Honda")
plt.loglog(energy_total, flux_nuMubar_honda_Juno, "g--", label="$\\bar{\\nu}_\\mu$-flux of Honda")
plt.loglog(energy_interesting, flux_nuE_JUNO, "r-", label="normalized $\\nu_e$-flux of FLUKA")
plt.loglog(energy_interesting, flux_nuEbar_JUNO, "k-", label="normalized $\\bar{\\nu}_e$-flux of FLUKA")
plt.loglog(energy_interesting, flux_nuMu_JUNO, "b-", label="normalized $\\nu_\\mu$-flux of FLUKA")
plt.loglog(energy_interesting, flux_nuMubar_JUNO, "g-", label="normalized $\\bar{\\nu}_\\mu$-flux of FLUKA")
plt.xlim(xmin=13.0, xmax=10000)
plt.xlabel("neutrino energy in MeV")
plt.ylabel("flux in 1/(MeV * s * cm$^2$)")
plt.title("Atmospheric neutrino flux at JUNO site")
plt.legend()
plt.grid()

plt.figure(2, figsize=(10, 5.5))
plt.loglog(energy_total, flux_nuE_honda_Juno, "r--", label="$\\nu_e$-flux of Honda")
plt.loglog(energy_total, flux_nuEbar_honda_Juno, "b--", label="$\\bar{\\nu}_e$-flux of Honda")
plt.loglog(energy_total, flux_nuMu_honda_Juno, "k--", label="$\\nu_\\mu$-flux of Honda")
plt.loglog(energy_total, flux_nuMubar_honda_Juno, "g--", label="$\\bar{\\nu}_\\mu$-flux of Honda")
plt.loglog(energy_FLUKA, flux_nuE_FLUKA_normalized, "r-", label="normalized $\\nu_e$-flux of FLUKA")
plt.loglog(energy_FLUKA, flux_nuEbar_FLUKA_normalized, "b-", label="normalized $\\bar{\\nu}_e$-flux of FLUKA")
plt.loglog(energy_FLUKA, flux_nuMu_FLUKA_normalized, "k-", label="normalized $\\nu_\\mu$-flux of FLUKA")
plt.loglog(energy_FLUKA, flux_nuMubar_FLUKA_normalized, "g-", label="normalized $\\bar{\\nu}_\\mu$-flux of FLUKA")
plt.xlim(xmin=13.0, xmax=10000)
plt.xlabel("neutrino energy in MeV")
plt.ylabel("flux in 1/(MeV * s * cm$^2$)")
plt.title("Atmospheric neutrino flux at JUNO site")
plt.legend()
plt.grid()

plt.show()

















