""" Script to analyze the data simulated with vis_spectrum.py:
    the simulated visible energies of signal and different background in JUNO from vis_spectrum.py
    are read and analyzed further """

import numpy as np
from matplotlib import pyplot

# input parameter:
# Dark matter mass in MeV (float):
mass_DM = 20.0
# energy corresponding to the electron-antineutrino energy in MeV (np.array of float64):
E1 = np.arange(10, 130, 0.01)
# energy corresponding to the visible energy in MeV, E2 defines the bins in pyplot.hist() (np.array of float64):
E2 = np.arange(8, 130, 0.1)
# exposure time in years (integer):
t_years = 10
number = 6000
number_vis = 1

# load the simulated data from the text-file (np.arrays of float):
TheoSpectrum_total = np.loadtxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}'
                                '_TheoSpectrum_1.txt'.format(mass_DM, E1[0], E1[-1]))
E_visible_signal = np.loadtxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}'
                              '_E_visible_signal_1.txt'.format(mass_DM, E1[0], E1[-1]))
E_visible_DSNB = np.loadtxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}'
                            '_E_visible_DSNB_1.txt'.format(mass_DM, E1[0], E1[-1]))
E_visible_CCatmospheric = np.loadtxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_'
                                     '{1:.0f}to{2:.0f}_E_visible_CCatmospheric_1.txt'.format(mass_DM, E1[0], E1[-1]))
E_visible_reactor = np.loadtxt('output_vis_spectrum/DMmass{0:.0f}_{1:.0f}to{2:.0f}/DMmass{0:.0f}_{1:.0f}to{2:.0f}'
                               '_E_visible_reactor_1.txt'.format(mass_DM, E1[0], E1[-1]))

# Total visible energy in MeV:
E_visible_total = np.append(E_visible_reactor, np.append(E_visible_CCatmospheric,
                                                         np.append(E_visible_signal, E_visible_DSNB)))


""" Display theoretical spectra with the settings below: """
# """
h1 = pyplot.figure(1)
pyplot.plot(E1, TheoSpectrum_total, 'k', label='total spectrum')
pyplot.xlim(E1[0], E1[-1])
pyplot.ylim(ymin=0)
pyplot.xticks(np.arange(10, E1[-1], 10))
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
pyplot.title("Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, mass_DM))
# """

""" Display theoretical spectrum with logarithmic y-scale: """
"""
h2 = pyplot.figure(2)
pyplot.semilogy(E1, TheoSpectrum_total, 'k', label='total spectrum')
pyplot.xlim(E1[0], E1[-1])
# pyplot.xticks(np.arange(4.0, E1[-1]), 2.0)
pyplot.xlabel("Electron-antineutrino energy in MeV")
pyplot.ylabel("Theoretical spectrum dN/dE in 1/MeV")
pyplot.title("Theoretical electron-antineutrino spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, mass_DM))
"""

""" Display visible spectrum: """
"""
h3 = pyplot.figure(1)
n_total, bins_total, patches_total = pyplot.hist(E_visible_total, bins=E2, histtype='step', color='k',
                                                 label='total spectrum')
n_signal, bins_signal, patches_signal = pyplot.hist(E_visible_signal, bins=E2, histtype='step', color='r',
                                                    label='signal from DM annihilation')
n_DSNB, bins_DSNB, patches_DSNB = pyplot.hist(E_visible_DSNB, bins=E2, histtype='step', color='b',
                                              label='DSNB background')
n_CCatmospheric, bins_CCatmospheric, patches_CCatmospheric = pyplot.hist(E_visible_CCatmospheric, bins=E2,
                                                                         histtype='step', color='g',
                                                                         label='atmospheric CC background '
                                                                               'without oscillation')
n_reactor, bins_reactor, patches_reactor = pyplot.hist(E_visible_reactor, bins=E2, histtype='step', color='c',
                                                       label='reactor electron-antineutrino background')
pyplot.xlim(E2[0], E2[-1])
pyplot.ylim(ymin=0, ymax=50000)
pyplot.xticks(np.arange(10.0, E2[-1], 10.0))
pyplot.xlabel("Visible energy in MeV")
pyplot.ylabel("Expected spectrum dN/dE in 1/MeV * {0:d}".format(number*number_vis))
pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, mass_DM))
pyplot.legend()
"""

""" Display visible spectrum with logarithmic y-scale: """
"""
h4 = pyplot.figure(2)
pyplot.hist(E_visible_total, bins=E2, histtype='step', color='k', log=True, label='total spectrum')
pyplot.hist(E_visible_signal, bins=E2, histtype='step', color='r', log=True, label='signal from DM annihilation')
pyplot.hist(E_visible_DSNB, bins=E2, histtype='step', color='b', log=True, label='DSNB background')
pyplot.hist(E_visible_CCatmospheric, bins=E2, histtype='step', color='g', log=True,
            label='atmospheric CC background without oscillation')
pyplot.hist(E_visible_reactor, bins=E2, histtype='step', color='c', log=True,
            label='reactor electron-antineutrino background')
pyplot.xlim(E2[0], E2[-1])
pyplot.ylim(ymin=0)
pyplot.xticks(np.arange(10.0, E2[-1], 10.0))
pyplot.xlabel("Visible energy in MeV")
pyplot.ylabel("Expected spectrum dN/dE in 1/MeV * {0:d}".format(number*number_vis))
pyplot.title("Expected spectrum in JUNO after {0:.0f} years and for DM of mass = {1:.0f} MeV"
             .format(t_years, mass_DM))
pyplot.legend()
"""

pyplot.show()
