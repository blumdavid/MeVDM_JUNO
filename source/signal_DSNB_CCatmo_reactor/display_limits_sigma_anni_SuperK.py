""" Script to display the different limits on the DM annihilation cross-section of Super-K and Hyper-K and to compare
    the different limits.

"""
import numpy as np
from matplotlib import pyplot as plt
import csv

""" 90% limit on DM self-annihilation cross-section from the whole Milky Way, obtained from Super-K Data (SK1).
    (results are digitized from figure 1, page 7 of paper 'Testing MeV Dark Matter with neutrino detectors', 
    arXiv: 0710.5420v1)
    The digitized data is saved in "/home/astro/blum/PhD/paper/MeV_DM/limit_SuperK_digitized.csv".
"""
# Dark matter mass in MeV (array of float):
m_dm_SuperK_old = np.array([13.3846, 13.5385, 13.6923, 13.6923, 14.1538, 15.2308, 15.8462, 16.7692, 17.6923, 19.0769,
                            20, 20.6154, 21.5385, 22.4615, 23.5385, 24.4615, 25.5385, 27.3846, 29.0769, 35.2308,
                            37.3846, 39.8462, 42.3077, 43.8462, 45.5385, 47.2308, 48.9231, 51.8462, 54.6154, 57.2308,
                            59.8462, 62.6154, 65.5385, 67.2308, 69.0769, 72.4615, 75.6923, 78.9231, 82, 84.9231,
                            87.6923, 90.1538, 92.6154, 95.6923, 98.7692, 104.154, 108.923, 113.385, 119.385, 120.769,
                            121.538, 122])
# 90% limit of the self-annihilation cross-section (np.array of float):
sigma_SuperK_old = np.array([4.74756E-23, 4.21697E-23, 2.7617E-23, 3.68279E-23, 1.08834E-23, 1.83953E-24, 7.62699E-25,
                             3.10919E-25, 1.63394E-25, 8.58665E-26, 6.43908E-26, 5.62341E-26, 4.99493E-26,
                             4.66786E-26, 4.51244E-26, 4.58949E-26, 4.82863E-26, 5.62341E-26, 6.54902E-26,
                             1.18448E-25, 1.50131E-25, 1.83953E-25, 2.17889E-25, 2.29242E-25, 2.33156E-25,
                             2.33156E-25, 2.25393E-25, 2.00203E-25, 1.63394E-25, 1.28912E-25, 1.03444E-25,
                             8.44249E-26, 7.49894E-26, 7.24927E-26, 7.24927E-26, 7.62699E-26, 8.44249E-26,
                             9.66705E-26, 1.14505E-25, 1.42696E-25, 1.77828E-25, 2.29242E-25, 3.00567E-25,
                             4.58949E-25, 7.12756E-25, 1.77828E-24, 4.66786E-24, 1.22528E-23, 4.99493E-23,
                             7.00791E-23, 8.30076E-23, 9.83212E-23])

""" 90 % limit on DM self-annihilation cross-section from whole Milky Way, obtained from Super-K data (SK1, SK2, SK3).
    (results are digitized from figure 1, page 4, of paper 'Dark Matter-neutrino interactions through the lens of their 
    cosmological implications', Phys.Rev.D97,075039)
    The digitized data is saved in '/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_new.csv'.
"""
m_dm_SuperK = []
sigma_SuperK = []

with open('/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_new.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        m_dm_SuperK.append(float(row[0]))
        sigma_SuperK.append(float(row[1]))

""" 90 % limit on DM self-annihilation cross-section from whole Milky Way, obtained from Super-K data (SK1, SK2, SK3).
    (results are digitized from figure 1, page 2, of paper 'Implications of a Dark Matter-Neutrino Coupling at 
    Hyper-Kamiokande', arXiv:1805.09830)
    The digitized data is saved in '/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_from_HyperK_plot.csv'.
    Should give the same limit like sigma_SuperK, because Phys.Rev.D97,075039 is the reference of the figure!
"""
m_dm_SuperK_fromHKplot = []
sigma_SuperK_fromHKplot = []

with open('/home/astro/blum/PhD/paper/MeV_DM/SuperK_limit_from_HyperK_plot.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        m_dm_SuperK_fromHKplot.append(float(row[0]))
        sigma_SuperK_fromHKplot.append(float(row[1]))

""" 90 % limit on DM self-annihilation cross-section from whole Milky Way, simulated for Hyper-K after 10 years of 
    measuring without Gd doping.
    (results are digitized from figure 1, page 2, of paper 'Implications of a Dark Matter-Neutrino Coupling at 
    Hyper-Kamiokande', arXiv:1805.09830)
    The digitized data is saved in '/home/astro/blum/PhD/paper/MeV_DM/HyperK_limit_no_Gd.csv'.
"""
m_dm_HyperK = []
sigma_HyperK = []

with open('/home/astro/blum/PhD/paper/MeV_DM/HyperK_limit_no_Gd.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        m_dm_HyperK.append(float(row[0]))
        sigma_HyperK.append(float(row[1]))


""" display all limits: """
plt.semilogy(m_dm_SuperK_old, sigma_SuperK_old, "k-", label="SuperK limit of 0710.5420")
plt.semilogy(m_dm_SuperK, sigma_SuperK, "b-", label="SuperK limit of Phys.Rev.D97,0750039")
plt.semilogy(m_dm_SuperK_fromHKplot, sigma_SuperK_fromHKplot, "r-", label="SuperK limit of 1805.09830\n"
                                                                          "(must be equal to blue line; just a cross-\n"
                                                                          "check for the digitization; red line is \n"
                                                                          "more precise -> better digitization tool)")
plt.semilogy(m_dm_HyperK, sigma_HyperK, "g-", label="HyperK limit (10 years)")
plt.xlim(xmin=0.0, xmax=200.0)
plt.xticks(np.arange(0, 200, 10))
plt.xlabel("Dark Matter mass $m_{DM}$ in MeV")
plt.ylabel("$<\\sigma_A v>$ in $cm^3/s$")
plt.title("90% limits on the averaged DM annihilation cross section as function of the DM mass")
plt.legend()
plt.grid()
plt.show()

