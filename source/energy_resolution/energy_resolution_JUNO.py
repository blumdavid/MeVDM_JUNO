# Script to calculate the energy resolution of the JUNO detector for different assumptions:
# Calculations are based on paper 1507.05613 (physics_report) from Juno collaboration

# energy resolution is defined as sigma/E_vis, but the width of the gaussian energy resolution function is given by
# sigma

import numpy as np
from matplotlib import pyplot as plt

# Define the visible energy in MeV:
E_vis = np.arange(1, 100.5, 0.5)

# simple estimation of the energy resolution:
# energy resolution sigma/E_vis in percent:
energy_res1 = 3 / np.sqrt(E_vis)
# width of the gaussian energy resolution function in MeV:
sigma1 = energy_res1 / 100 * E_vis

# generic parametrization of the energy resolution (from physics_report: p. 195 table 13-4 and page 45):
# parameter dominated by the photon statistics:
p0 = 2.8
# parameters, that come from detector effects such as PMT dark noise, variation of the PMT QE and
# the reconstructed vertex smearing:
p1 = 0.26
p2 = 0.90
# energy resolution sigma/E_vis in percent:
energy_res2 = np.sqrt((p0/np.sqrt(E_vis))**2 + p1**2 + (p2/E_vis)**2)
# width of the gaussian energy resolution function in MeV:
sigma2 = energy_res2 / 100 * E_vis

print(energy_res2)

plt.figure(1)
plt.plot(E_vis, energy_res1, label='simple approach: 3 % / '+'$\sqrt{E_{vis}}$')
plt.plot(E_vis, energy_res2, label='generic parametrization: a=2.8, b=0.26, c=0.9')
plt.xlim(E_vis[0], E_vis[-1])
plt.ylim(ymin=0)
plt.xticks(np.arange(0, 105, 5))
# plt.yticks(np.arange(0, 3.5, 0.25))
plt.xlabel("Visible energy in MeV")
plt.ylabel("energy resolution in %")
plt.title("Energy resolution of the JUNO detector")
plt.grid()
plt.legend()

plt.figure(2)
plt.plot(E_vis, sigma1, label='simple approach')
plt.plot(E_vis, sigma2, label='generic parametrization: p0=2.8, p1=0.26, p2=0.9')
plt.xlim(E_vis[0], E_vis[-1])
plt.ylim(ymin=0)
plt.xticks(np.arange(0, 130, 10))
# plt.yticks(np.arange(0, max(sigma2), 2.5))
plt.xlabel("Visible energy in MeV")
plt.ylabel("Width of the gaussian energy resolution sigma in MeV")
plt.title("Width of the gaussian energy resolution sigma of the JUNO detector")
plt.grid(True)
plt.legend()

plt.show()
