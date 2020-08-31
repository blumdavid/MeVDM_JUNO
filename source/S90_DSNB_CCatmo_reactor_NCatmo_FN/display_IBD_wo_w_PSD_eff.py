""" script to display the simulated IBD spectrum before PSD and after PSD.

    Also the PSD efficiency is displayed.

"""
import numpy as np
from matplotlib import  pyplot as plt

path_PSD_info = "/home/astro/blum/juno/atmoNC/data_NC/output_detsim_v2/" \
                "DCR_results_16000mm_10MeVto100MeV_1000nsto1ms_mult1_1800keVto2550keV_dist500mm_R17700mm_PSD99/" \
                "test_10to20_20to30_30to40_40to100_final/"

# set energy window in MeV:
energy_min = 10.0
energy_max = 100.0
energy_interval_IBD = 1.0
energy_interval_NC = 0.5
energy_IBD = np.arange(energy_min, energy_max+energy_interval_IBD, energy_interval_IBD)
energy_NC = np.arange(energy_min, energy_max+energy_interval_NC, energy_interval_NC)

""" PSD Efficiency of IBD events: """
# total PSD suppression for real IBD events (see folder path_PSD_info):
PSD_supp_IBD_total = 0.1160
# total PSD efficiency in percent:
PSD_eff_IBD_total = (1.0 - PSD_supp_IBD_total) * 100.0
# load simulated IBD spectrum before PSD (from 10 to 100 MeV with bin-width 1 MeV):
array_IBD_spectrum = np.loadtxt(path_PSD_info + "IBDspectrum_woPSD_bin1000keV.txt")
# load simulated IBD spectrum after PSD (from 10 to 100 MeV with bin-width 1 MeV):
array_IBD_spectrum_PSD = np.loadtxt(path_PSD_info + "IBDspectrum_wPSD_bin1000keV.txt")

# calculate PSD survival efficiency of IBD events for each energy bin (bin-width 1 MeV) in percent:
# INFO-me: survival efficiency and NOT suppression is calculated!
array_eff_IBD_1MeV = array_IBD_spectrum_PSD / array_IBD_spectrum * 100.0
# INFO-me: last entry of array_IBD_spectrum = 0, therefore is the last entry of array_eff_IBD_1MeV = NaN!
# Therefore replay the last entry of array_eff_IBD_1MeV by the second last value:
array_eff_IBD_1MeV[-1] = array_eff_IBD_1MeV[-2]


""" PSD efficiency of NC events: """
# total PSD suppression of atmospheric NC events:
PSD_supp_NC_total = 0.9912
# total PSD efficiency in percent:
PSD_eff_NC_total = (1.0 - PSD_supp_NC_total) * 100.0
# load simulated NC spectrum before PSD (from 10 to 100 MeV with bin-width 0.5 MeV):
array_NC_spectrum = np.loadtxt(path_PSD_info + "NCatmo_onlyC12_woPSD_bin500keV.txt")
# load simulated NC spectrum after PSD (from 10 to 100 MeV with bin-width 0.5 MeV):
array_NC_spectrum_PSD = np.loadtxt(path_PSD_info + "NCatmo_onlyC12_wPSD99_bin500keV.txt")

# calculate PSD survival efficiency of NC events for each energy bin (bin-width 0.5 MeV) in percent:
array_eff_NC_500keV = array_NC_spectrum_PSD / array_NC_spectrum * 100.0

""" PSD efficiency of FN events: """
# see fast_neutron_summary.ods in folder /home/astro/blum/PhD/work/MeVDM_JUNO/fast_neutrons/).
# Fast neutron efficiency and IBD efficiency depend both on NC efficiency and on the energy.
# total PSD suppression for fast neutron events
PSD_supp_FN_total = 0.9994
# total PSD efficiency for FN events:
PSD_eff_FN_total = (1.0 - PSD_supp_FN_total) * 100.0
# PSD survival efficiency of fast neutron events in percent:
PSD_eff_FN_10_20 = (1.0 - 1.0) * 100.0
PSD_eff_FN_20_30 = (1.0 - 0.9984) * 100.0
PSD_eff_FN_30_40 = (1.0 - 0.9976) * 100.0
PSD_eff_FN_40_100 = (1.0 - 0.9997) * 100.0

# preallocate array where PSD efficiency of FN events are stored:
array_eff_FN_1MeV = np.zeros(len(energy_IBD))
for index in range(len(array_eff_FN_1MeV)):
    if index < 10:
        array_eff_FN_1MeV[index] = PSD_eff_FN_10_20
    elif 10 <= index < 20:
        array_eff_FN_1MeV[index] = PSD_eff_FN_20_30
    elif 20 <= index < 30:
        array_eff_FN_1MeV[index] = PSD_eff_FN_30_40
    else:
        array_eff_FN_1MeV[index] = PSD_eff_FN_40_100

""" display efficiencies: """
h1 = plt.figure(1, figsize=(9, 6))
plt.step(energy_IBD, array_eff_IBD_1MeV, "r-",
         label="IBD events ($\\epsilon_{IBD}=$" + " {0:.2f} %)".format(PSD_eff_IBD_total))
plt.step(energy_NC, array_eff_NC_500keV, color="orange", linestyle="solid",
         label="atmo. NC events ($\\epsilon_{atmoNC}=$" + " {0:.2f} %)".format(PSD_eff_NC_total))
plt.step(energy_IBD, array_eff_FN_1MeV, color="magenta", linestyle="solid",
         label="FN events ($\\epsilon_{FN}=$" + " {0:.2f} %)".format(PSD_eff_FN_total))
plt.xlim(energy_min, energy_max)
plt.ylim(0.0, 110.0)
plt.xlabel('Visible energy $E_{vis}$ in MeV', fontsize=12)
plt.ylabel("PSD efficiency in %")
plt.title("PSD efficiencies of IBD, atmo. NC and FN events")
plt.grid()
plt.legend()

h2 = plt.figure(2, figsize=(9, 6))
plt.semilogy(energy_IBD, array_eff_IBD_1MeV, "r-", drawstyle="steps",
             label="IBD events ($\\epsilon_{IBD}=$" + " {0:.2f} %)".format(PSD_eff_IBD_total))
plt.semilogy(energy_NC, array_eff_NC_500keV, color="orange", linestyle="solid", drawstyle="steps",
             label="atmo. NC events ($\\epsilon_{atmoNC}=$" + " {0:.2f} %)".format(PSD_eff_NC_total))
plt.semilogy(energy_IBD, array_eff_FN_1MeV, color="magenta", linestyle="solid", drawstyle="steps",
             label="FN events ($\\epsilon_{FN}=$" + " {0:.2f} %)".format(PSD_eff_FN_total))
plt.xlim(energy_min, energy_max)
plt.ylim(0.0, 110.0)
plt.xlabel('Visible energy $E_{vis}$ in MeV', fontsize=12)
plt.ylabel("PSD efficiency in %")
plt.title("PSD efficiencies of IBD, atmo. NC and FN events")
plt.grid()
plt.legend()

""" display simulated IBD spectrum without and with PSD: """
h3 = plt.figure(3, figsize=(9, 6))
plt.step(energy_IBD, array_IBD_spectrum, linestyle="-", color="red",
         label="before PSD: number of events = {0:.0f}".format(np.sum(array_IBD_spectrum)))
# plt.fill_between(energy_IBD, np.zeros(len(array_IBD_spectrum)), array_IBD_spectrum,
#                  color="red", alpha=0.8, step='pre')
# plt.step(energy_IBD, array_IBD_spectrum_PSD, linestyle="-", color="red")
plt.fill_between(energy_IBD, np.zeros(len(array_IBD_spectrum_PSD)), array_IBD_spectrum_PSD,
                 color="red", alpha=0.6, step='pre',
                 label="after PSD: number of events = {0:.0f}".format(np.sum(array_IBD_spectrum_PSD)) +
                       ", $\\Sigma_{IBD}$" + " = {0:.2f} %".format(PSD_supp_IBD_total))
plt.xlim(xmin=energy_min, xmax=energy_max)
plt.ylim(ymin=0.0)
plt.xlabel("Visible energy in MeV", fontsize=12)
plt.ylabel("number of events per bin (bin-width = {0:.1f} MeV)".format(energy_interval_IBD), fontsize=12)
plt.title("Spectra of simulated IBD events that pass the IBD selection criteria\n"
          "(before and after pulse shape discrimination)")
plt.legend(loc='lower right')
plt.grid()

fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

color = 'r'
ax1.set_xlabel("Visible energy in MeV", fontsize=12)
ax1.set_ylabel("number of events per bin (bin-width = {0:.1f} MeV)".format(energy_interval_IBD), color=color, fontsize=12)
ax1.step(energy_IBD, array_IBD_spectrum, linestyle="-", color=color,
         label="before PSD: number of events = {0:.0f}".format(np.sum(array_IBD_spectrum)))
ax1.fill_between(energy_IBD, np.zeros(len(array_IBD_spectrum_PSD)), array_IBD_spectrum_PSD,
                 color=color, alpha=0.6, step='pre',
                 label="after PSD: number of events = {0:.0f}".format(np.sum(array_IBD_spectrum_PSD)) +
                       ", $\\Sigma_{IBD}$" + " = {0:.2f} %".format(PSD_supp_IBD_total))
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlim(energy_min, energy_max)
ax1.xaxis.grid()
ax1.set_ylim(ymin=0.0)
ax1.set_title("Spectra of simulated IBD events (before and after PSD)\n"
              "and corresponding PSD efficiency", fontsize=13)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'k'
ax2.set_ylabel('PSD efficiency $\\epsilon_{PSD,IBD}$ in %', color=color, fontsize=12)
ax2.step(energy_IBD, array_eff_IBD_1MeV, color=color,
         label="$\\epsilon_{IBD}$")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.0, 200.0)
ax2.set_yticks(np.arange(0, 120, 20))
ax2.grid()

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()












