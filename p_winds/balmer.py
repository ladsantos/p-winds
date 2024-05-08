#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module computes the neutral and ionized populations of H in the upper
atmosphere. It is designed for the Balmer series of hydrogen.
"""

import numpy as np
import astropy.units as u
import astropy.constants as const
import sys

def calc_boltzmann_distribution(T, n, NLTE_scaling = 1.):
    """
    Calculates the distribution of electronic states of the different atomic shells
    of the hydrogen atom via the Boltzmann equation in LTE and NLTE.
    In short, it calculates which fraction of the total number of H atoms are in the shell needed to produce H-alpha, H-beta etc lines.

    Parameters
    ----------
    T: ``float``
        Temperature in K
    
    n : ``integer``
        For which shell number do we calculate the Boltzmann distribution

    NLTE_scaling : ``float``
        Default: 1. (LTE, no scaling)
        If NLTE is assumed, the scaling factor can be a free fitting parameter in the retrieval on the data

    Returns
    -------
    boltzmann_n : ``float``
        Unitless fraction of Hydrogen population in the needed atomic shell for the specific Balmer-line

    """
        
    # Energy of the nth state in the Balmer series as a difference to the ground state
    #ground state energy for hydrogen is 13.6 eV
    #E_n = E1 - E(n) in our case with E(n) = E1/n**2 for the Balmer series
    E_n = 13.6 * (1 - 1/n**2) * u.eV
    
    # Population distribution via the Boltzmann equation
    #This is the limit case for a large population of atoms as applicable in atmospheres
    #Boltzmann_n = g_n/g_1 * np.exp(-E_n / (const.k_B * T))
    #g_1 in our case is a constant and always 2
    #g_n are the statistical weights of the electronic levels due to their degeneracies
    #to generalise this function for other elements, simply let the user provide these as input, as well as the energy difference
    g_n = 2.*n**2.
    g_1 = 2.
    
    #The NLTE scaling is defined as NLTE_scaling = Boltzmann_n (NLTE) / Boltzmann_n (LTE)
    #from Barman et al. 2002 and others
    #The scaling factor can be assumed as a constant for thermospheres
    #from Huang et al. 2017, and Garcia Munoz and Schneider (2019)
    
    #calculate fraction of level population
    boltzmann_n = NLTE_scaling * (g_n/g_1) * np.exp(-E_n / (const.k_B * T)).decompose()
        
    
    return boltzmann_n

def calc_saha_distribution(T, electron_density, n):
    """
    Calculate the distribution of ionization states for the Balmer series of hydrogen
    using the Saha equation.

    Parameters
    ----------
    T: ``float``
        Temperature in K
        
    electron_density: ``float``
        the electron density (most likely coming from the background, but tbd). If only one level of ionisation is important then n1 = ne
        The electron density has to have units of (cm**-3)
    
    n : ``integer``
        Number of the atomic shell

    Returns
    -------
    saha_n : ``float``
        Unitless fraction of ionised Hydrogen from total Hydrogen for the specific Balmer-line
    """

    
    # Boltzmann constant in eV/K
    k_b = const.k_B.to(u.eV / u.K)

    # Ionization potential of hydrogen (eV)
    ionization_potential = 13.6 * u.eV

    # Partition function for hydrogen
    #is the sum over all quantum states for the atom where the electron could be, thus the sum over g_n*exp(-(E1-En)/(kb*T)
    # in the case of hydrogen any contributions beyond n=2 are negligible
    U_i = 2. + 8.*np.exp(-3.4 * u.eV / (k_b * T))
    
    #thermal deBroglie wavelength ** (-1)
    debroglie_rev = np.sqrt(2 * np.pi * const.m_e * k_b * T / (const.hbar**2))

    # Population distribution
    #in reality the equation is not divided by U_i but multliplied by U_i(H II)/U_i(H I)
    #For the Balmer series, H II is the naked proton core, as hydrogen only has one electron. Therefore, U_i(H II) = 1
    saha_n = 2./(electron_density * U_i )* (debroglie_rev)**(3) * np.exp(-ionization_potential * (1 - 1/n**2) / (k_b * T))

    return saha_n
    
def relation_boltzmann_saha(boltzmann_frac, saha_frac):

    #relates the boltzmann and saha distribution together via the total number of hydrogen atoms
    total_fraction = boltzmann_frac/((1+boltzmann_frac)*(1+saha_frac))

    return total_fraction
    
def calc_opacity(T, line , NLTE=1., Voigt)
    """
    Calculates the opacity for a specific Balmer line at a specific temperature considering the initial Voigt profile as input.
    Technically the Lorenzian wings of the Voigt profile should take the Einstein coefficients of the Balmer series into account.
    This is not yet implemented. Einstein coefficients: log10(A n2 [s−1]) = 8.76, 8.78, 8.79 for n = 3, 4, and 5
    Afterwards this opacity has to be used to calculate tau in the line of sight. I assume this is implemented in p-winds

    Parameters
    ----------
    T: ``float``
        Temperature in K
        
    line : ``string``
        Default: ``alpha`` for the H-alpha line
        Which line of the Balmer series is computed. Accepted values are 'alpha', 'beta', 'gamma', and 'epsilon' for the lines accessible in the optical.
        
    NLTE_scaling : ``float``
        Default: 1. (LTE, no scaling)
        If NLTE is assumed, the scaling factor can be a free fitting parameter in the retrieval on the data
    
    Voigt:
        implementation to be determined

    Returns
    -------
    opacity2n: ``float``
        Opacity of the Balmer line taking into account ionisation via the Saha equation.
    """

    if line == 'alpha':
        n = 3
        gf = 10**(0.71)
    elif line == 'beta':
        n = 4
        gf = 10**(-0.02)
    elif line == 'gamma':
        n = 5
        gf = 10**(-0.447)
    else:
        #This is a placeholder and should be implemented upstream via the proper input testing module of p-winds
        print("The spectral line you specified is not yet implemented. Accepted values are: alpha, beta, gamma.")
        sys.exit()
        
    #this is the ground level ionisation
    saha_2 = calc_saha_distribution(T,2)
    
    #this is the ground level distribution
    boltzmann_2 = calc_boltzmann_distribution(T, 2, NLTE)
    
    #this is the upper level ionisation, depending on the line
    saha_n = calc_saha_distribution(T,n)
    
    #this is the upper level distribution, depending on the line
    boltzmann_ion = calc_boltzmann_distribution(T, n, NLTE)
    
    
    #relate Boltzmann distribution to Saha distribution
    base_fraction = relation_boltzmann_saha(boltzmann_2, saha_2)
    upper_fraction = relation_boltzmann_saha(boltzmann_n, saha_n)
    
    #equation 9 from Wyttenbach et al. 2020 adjusted for Saha
    #we use 2.01*const.u for the mass of hydrogen
    opacity2n = np.pi*const.e**2./(const.m_e*const.c) * gf/(2.01*const.u) * (base_fraction/8. - upper_fraction/(2.*n**2.)*Voigt)

    return opacity2n

