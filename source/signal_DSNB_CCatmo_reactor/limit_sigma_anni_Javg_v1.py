""" Script to calculate the 90 percent dark matter self-annihilation cross-section for different dark matter masses
    and different values of the angular-averaged DM intensity J_avg over the whole Milky Way and display the results.

    Results of the simulation and analysis saved in folder "S90_DSNB_CCatmo_reactor".

    Information about the expected signal and background spectra generated with gen_spectrum_v2.py:

    Background: - reactor electron-antineutrinos background
                - DSNB
                - atmospheric Charged Current electron-antineutrino background

    Diffuse Supernova Neutrino Background:
    - expected spectrum of DSNB is saved in file: DSNB_EmeanNuXbar22_bin100keV.txt
    - information about the expected spectrum is saved in file: DSNB_info_EmeanNuXbar22_100keV.txt

    Reactor electron-antineutrino Background:
    - expected spectrum of reactor background is saved in file: Reactor_NH_power36_bin100keV.txt
    - information about the reactor background spectrum is saved in file: Reactor_info_NH_power36_bin100keV.txt

    Atmospheric Charged Current electron-antineutrino Background:
    - expected spectrum of atmospheric CC background is saved in file: CCatmo_Osc1_bin100keV.txt
    - information about the atmospheric CC background spectrum is saved in file: CCatmo_info_Osc1_bin100keV.txt

    DM annihilation cross-section as function of DM mass depending on J_avg
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_annihilation_crosssection
from work.MeVDM_JUNO.source.gen_spectrum_functions import limit_neutrino_flux