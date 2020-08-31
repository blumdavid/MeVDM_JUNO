""" Script to analyze the reactor electron-antineutrino spectrum from Fallot2012_PhysRevLett.109.202504:
    - The question was, if you have to take the binning of the spectra in figure 1 into account.
    - Therefore several calculations and considerations have been made (see hand writen notes).
    - Here the calculation are done for the data fo U235 of figure 1 as an example.
    - Result:
    The calculation of the average number of neutrinos per fission do NOT depend on the chosen binning.
    The reason is, that the flux is given in neutrinos/MeV/fission and therefore the binning is considered (unit 1/MeV).

    -> reactor neutrino flux of Fallot do NOT depend on the binning!!!!!
"""

import numpy as np
from matplotlib import pyplot

# bin-width in figure 1 in MeV (=100keV):
binwidth_fig = 0.1

# energy points from digitization in MeV (one data-point of 17 MeV is added, important for interpolation):
energy_digi = np.array([0.3841, 0.7464, 1.1087, 1.471, 1.7609, 2.1232, 2.4855, 2.8478, 3.1377, 3.5, 3.8623,
                        4.1522, 4.5145, 4.8043, 5.0942, 5.3841, 5.7464, 6.0362, 6.3261, 6.6159, 6.9058, 7.1957,
                        7.4855, 7.7754, 7.9928, 8.1377, 8.3551, 8.6449, 8.8623, 9.1522, 9.442, 9.7319, 9.8768,
                        10.1667, 10.529, 10.8188, 11.1087, 11.3986, 11.6884, 11.8333, 12.0507, 12.1232, 12.1957,
                        12.2681, 12.3406, 12.4855, 12.7754, 13.1377, 13.3551, 13.5725, 13.9348, 14.2246, 14.442,
                        14.7319, 14.9493, 15.0217, 15.0942, 15.1667, 15.3841, 15.6739, 15.8913, 16.1087, 17.00])

# electron-antineutrino flux from figure 1 in antineutrinos/(MeV*fission) (data-point of 0.0 events/(MeV*fission) for
# 17 MeV ia added):
# values to the energy points in energy_digi:
flux_fig = np.array([1.548, 1.797, 1.797, 1.548, 1.334, 1.149, 0.852, 0.734, 0.5446, 0.4041, 0.3481, 0.2224,
                     0.165, 0.1225, 0.0909, 0.06741, 0.05001, 0.03196, 0.02371, 0.01759, 0.01124, 0.007186,
                     0.004592, 0.002528, 0.001392, 0.000766, 0.0003632, 0.0002321, 0.0001484, 0.0000948,
                     0.00006059, 0.00003872, 0.00002132, 0.00001362, 0.00001011, 0.000006459, 0.000004792,
                     0.000003063, 0.000001686, 0.000000928, 0.0000005931, 0.0000002812, 0.0000001334,
                     0.00000006323, 0.00000002998, 0.0000000165, 0.00000001225, 0.00000000783,
                     0.000000005001, 0.000000003196, 0.000000002043, 0.000000001515, 0.000000000969,
                     0.000000000619, 0.0000000003407, 0.0000000001616, 0.00000000003632, 0.00000000001722,
                     0.00000000001101, 0.00000000000817, 0.000000000004495, 0.000000000002475, 0.0])

# number of neutrinos per fission from digitization-points:
# determine bin-width of energy_digi:
binwidth_digi = np.array([])
for index in np.arange(len(energy_digi[0:-1])):
    binwidth_digi = np.append(binwidth_digi, energy_digi[index+1] - energy_digi[index])
# flux for values of digitization (bin-values is defines by the value on the left side of the bin,
# therefore the last entry in flux_fig is neglected):
flux_digi = flux_fig[0:-1]
# sum over product of flux_digi and binwidth_digi to get the number of antineutrinos per fission:
number_digi = sum(flux_digi*binwidth_digi)
print("average number of electron-antineutrinos per fission:\nnumber_digi = {0:.3f} neutrinos/fission"
      .format(number_digi))


# number of neutrinos per fission from interpolation of the data from digitization:
# binwidth of interpolation in MeV:
binwidth_inter = np.array([1, 0.25, 0.1, 0.05, 0.01, 0.005])
# loop over different binwidth of interpolation:
for index in np.arange(0, 6):
    # energy points from interpolation can be defined with the given binwidth_inter, in MeV:
    energy_inter = np.arange(1.8, energy_digi[-1], binwidth_inter[index])
    # flux from figure 1 to the energy points in energy_inter:
    flux_inter = np.interp(energy_inter, energy_digi, flux_fig)
    # sum over product of flux_inter and binwidth_inter to get the number of antineutrinos per fission:
    number_inter = sum(flux_inter) * binwidth_inter[index]
    print("average number of electron-antineutrinos per fission for bin-width {0:.3f} MeV:\n"
          "number_inter = {1:.3f} neutrinos/fission".format(binwidth_inter[index], number_inter))

""" Display the digitized and interpolated data: """
pyplot.plot(energy_inter, flux_inter, '-', label='interpolated')
pyplot.plot(energy_digi, flux_fig, '+', label='digitized')
pyplot.legend()
pyplot.grid()
pyplot.show()
