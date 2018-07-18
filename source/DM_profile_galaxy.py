""" Script to display the different Dark Matter profiles in the galaxy (DM_profile_galaxy.py):

    Source: 0710.5420_MeVDMMilkyWaySKLena.pdf

    Stable particles with masses in the MeV range constitute a cold DM candidate. Detailed structure formation
    simulations show that cold DM clusters hierarchically in halos and the formation of large scale structure in the
    Universe can be successfully reproduced.
    In the case of spherically symmetric matter density with isotropic velocity dispersion, the simulated DM profile
    in the galaxies can be parametrized.

    While DM profiles tend to agree at large scales, uncertainties are still present for the inner region of the galaxy.
    As the neutrino flux from DM annihilation scales as rho**2 ('DM profile squared'), this leads to an uncertainty in
    the overall normalization of the flux.
    To understand this quantitatively, we have studied the impact of the chosen halo profile by choosing three
    spherically symmetric profiles with isotropic velocity dispersion (from more to less cuspy).

"""
import numpy as np
from matplotlib import pyplot as plt


def dm_profile_parametrization(r, rho_sc, r_s, gamma, alpha, beta):
    """
    In the case of spherically symmetric matter density with isotropic velocity dispersion, the simulated DM profile
    in the galaxies can be parametrized with this function.

    The equation of the function is from the paper on page 2, equation (1).

    :param r: distance to the center of the Milky Way (in units of kpc) (np.array of float)
    :param rho_sc: DM density at r_sc (in units of GeV/cm**3) (float)
    :param r_s: scale radius (in units of kpc) (float)
    :param gamma: inner cusp index (float)
    :param alpha: determines the exact shape of the profile in the regions around r_s (float)
    :param beta: slope as r -> infinity (float)

    :return: dm_profile: DM profile density in the Milky Way as function of the distance to the center of the Milky Way
                         (in units of GeV/cm**3) (np.array of float)
    """
    # define the solar radius circle in kpc ('distance of the solar orbit to the center of the Milky Way) (float):
    r_sc = 8.5

    # calculate the DM profile density from equation (1) of the paper (units: GeV/cm**3) (np.array of float):
    dm_profile = rho_sc * (r_sc / r)**gamma * ((1 + (r_sc/r_s)**alpha) / (1 + (r/r_s)**alpha))**((beta-gamma)/alpha)

    return dm_profile


""" define the array of the distance from the center of the Milky Way in kpc (np.array of float): """
R = np.arange(0.5, 30, 0.1)


""" calculate the DM profile of the Milky Way for the MQGSL (Moore, Quinn, Governato, Stadel, Lake) model: """
# exact shape of the profile in the regions around R_s for MQGSL model (float):
alpha_MQGSL = 1.5
# slope as R -> infinity for MQGSL model (float):
beta_MQGSL = 3
# inner cusp index for MQGSL model (float):
gamma_MQGSL = 1.5
# scale radius in kpc for MQGSL model (float):
r_s_MQGSL = 28
# minimum and maximum of the DM density at r_sc for the MQGSL model, which satisfy the present constraints from the
# allowed range for the local rotational velocity, the amount of flatness of the rotational curve of the Milky Way and
# the maximal amount of its non-halo components (units: GeV/cm**3) (float):
rho_sc_min_MQGSL = 0.22
rho_sc_max_MQGSL = 0.98

# DM profile in GeV/cm**3 for the MQGSL model (for minimum and maximum of the DM density at r_sc) (np.array of float):
DM_profile_min_MQGSL = dm_profile_parametrization(R, rho_sc_min_MQGSL, r_s_MQGSL, gamma_MQGSL, alpha_MQGSL, beta_MQGSL)
DM_profile_max_MQGSL = dm_profile_parametrization(R, rho_sc_max_MQGSL, r_s_MQGSL, gamma_MQGSL, alpha_MQGSL, beta_MQGSL)


""" calculate the DM profile of the Milky Way for the NFW (Navarro, Frenk, White) model: """
# exact shape of the profile in the regions around R_s for NFW model (float):
alpha_NFW = 1
# slope as R -> infinity for NFW model (float):
beta_NFW = 3
# inner cusp index for NFW model (float):
gamma_NFW = 1
# scale radius in kpc for NFW model (float):
r_s_NFW = 20
# minimum and maximum of the DM density at r_sc for the NFW model, which satisfy the present constraints from the
# allowed range for the local rotational velocity, the amount of flatness of the rotational curve of the Milky Way and
# the maximal amount of its non-halo components (units: GeV/cm**3) (float):
rho_sc_min_NFW = 0.20
rho_sc_max_NFW = 1.11

# DM profile in GeV/cm**3 for the NFW model (for minimum and maximum of the DM density at r_sc) (np.array of float):
DM_profile_min_NFW = dm_profile_parametrization(R, rho_sc_min_NFW, r_s_NFW, gamma_NFW, alpha_NFW, beta_NFW)
DM_profile_max_NFW = dm_profile_parametrization(R, rho_sc_max_NFW, r_s_NFW, gamma_NFW, alpha_NFW, beta_NFW)


""" calculate the DM profile of the Milky Way for the KKBP (Kravstov, Klypin, Bullock, Primack) model: """
# exact shape of the profile in the regions around R_s for KKBP model (float):
alpha_KKBP = 2
# slope as R -> infinity for KKBP model (float):
beta_KKBP = 3
# inner cusp index for KKBP model (float):
gamma_KKBP = 0.4
# scale radius in kpc for KKBP model (float):
r_s_KKBP = 10
# minimum and maximum of the DM density at r_sc for the KKBP model, which satisfy the present constraints from the
# allowed range for the local rotational velocity, the amount of flatness of the rotational curve of the Milky Way and
# the maximal amount of its non-halo components (units: GeV/cm**3) (float):
rho_sc_min_KKBP = 0.32
rho_sc_max_KKBP = 1.37

# DM profile in GeV/cm**3 for the KKBP model (for minimum and maximum of the DM density at r_sc) (np.array of float):
DM_profile_min_KKBP = dm_profile_parametrization(R, rho_sc_min_KKBP, r_s_KKBP, gamma_KKBP, alpha_KKBP, beta_KKBP)
DM_profile_max_KKBP = dm_profile_parametrization(R, rho_sc_max_KKBP, r_s_KKBP, gamma_KKBP, alpha_KKBP, beta_KKBP)


""" display the profiles in a plot: """
h1 = plt.figure(1, figsize=(15, 8))
plt.plot(R, DM_profile_min_MQGSL, 'r--', label='MQGSL model for $(\\rho_{sc})_{min,MQGSL}$')
plt.plot(R, DM_profile_max_MQGSL, 'r-', label='MQGSL model for $(\\rho_{sc})_{max,MQGSL}$')
plt.fill_between(R, DM_profile_min_MQGSL, DM_profile_max_MQGSL, facecolor='red', alpha=0.3)
plt.plot(R, DM_profile_min_NFW, 'b--', label='NFW model for $(\\rho_{sc})_{min,NFW}$')
plt.plot(R, DM_profile_max_NFW, 'b-', label='NFW model for $(\\rho_{sc})_{max,NFW}$')
plt.fill_between(R, DM_profile_min_NFW, DM_profile_max_NFW, facecolor='blue', alpha=0.3)
plt.plot(R, DM_profile_min_KKBP, 'g--', label='KKBP model for $(\\rho_{sc})_{min,KKBP}$')
plt.plot(R, DM_profile_max_KKBP, 'g-', label='KKBP model for $(\\rho_{sc})_{max,KKBP}$')
plt.fill_between(R, DM_profile_min_KKBP, DM_profile_max_KKBP, facecolor='green', alpha=0.3)
plt.xlabel('distance R to the center of the Milky Way in kpc')
plt.ylabel('DM density $\\rho(r)$ in $GeV/cm^3$')
plt.title('DM profiles for the Milky Way and different DM models')
plt.legend()
plt.grid()
plt.show()
