#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the p-winds wrapper code to run the fluid hydrodynamics
models of ATES.

ATES was originally developed by Andrea Caldiroli. For p-winds, we utilize a
custom version of ATES (forked from the original) that can be downloaded at the
following link: https://github.com/ladsantos/ATES-Code. For the original
version, go to https://github.com/AndreaCaldiroli/ATES-Code, but note that this
module only works with the custom version.
"""

import os
import subprocess
import glob

import numpy as np

from scipy.integrate import trapezoid
from warnings import warn


__all__ = ["ates_model", "run_ates"]


# Find $ATES_DIR environment variable
try:
    _ATES_DIR = os.environ["ATES_DIR"]
except KeyError:
    _ATES_DIR = None
    warn("Environment variable ATES_DIR is not set.")


# General warning
warn("This code is in development and not ready for production yet.")


# Run the ATES code using a Python wrapper
def ates_model(planet_radius, planet_mass, planet_equilibrium_temperature,
               semi_major_axis, stellar_mass, stellar_spectrum,
               he_h_ratio=0.1111, escape_radius=2.0, log_base_density=14.0,
               momentum_threshold=0.05, compiler='gfortran'):

    # Open input file and clean contents if file exists
    input_file = "input.inp"
    open(input_file, 'w').close()

    # X-rays wavelength range is 10-100 Angstrom
    xray_lim = [10, 100]
    # Extreme-UV wavelength range is 100-921 Angstrom
    euv_lim = [100, 921]

    # Read stellar spectrum and calculate luminosity in the X-rays and EUV parts
    au_to_cm = 1.49597871E+13
    flux_to_lum = 4 * np.pi * (semi_major_axis * au_to_cm) ** 2
    wavelength = stellar_spectrum['wavelength']  # Angstrom
    flux_density = stellar_spectrum['flux_lambda']  # erg / s / cm ** 2 / AA

    # Define the ranges in index space
    wavelength_xray = wavelength[
        np.bitwise_and(wavelength > xray_lim[0], wavelength < xray_lim[1])
    ]
    flux_density_xray = flux_density[
        np.bitwise_and(wavelength > xray_lim[0], wavelength < xray_lim[1])
    ]
    wavelength_euv = wavelength[
        np.bitwise_and(wavelength > euv_lim[0], wavelength < euv_lim[1])
    ]
    flux_density_euv = flux_density[
        np.bitwise_and(wavelength > euv_lim[0], wavelength < euv_lim[1])
    ]

    # Calculate luminosities
    lum_xray = trapezoid(flux_density_xray, wavelength_xray) * flux_to_lum
    lum_euv = trapezoid(flux_density_euv, wavelength_euv) * flux_to_lum
    log_lx = np.log10(lum_xray)
    log_leuv = np.log10(lum_euv)

    input_strings = [
        "Planet name: Undefined",
        "\nLog10 lower boundary number density [cm^-3]: " + str(
            log_base_density),
        "\nPlanet radius [R_J]: " + str(planet_radius),
        "\nPlanet mass [M_J]: " + str(planet_mass),
        "\nEquilibrium temperature [K]: " + str(planet_equilibrium_temperature),
        "\nOrbital distance [AU]: " + str(semi_major_axis),
        "\nEscape radius [R_p]: " + str(escape_radius),
        "\nHe/H number ratio: " + str(he_h_ratio),
        "\n2D approximate method: Rate/2 + Mdot/2",
        "\nParent star mass [M_sun]: " + str(stellar_mass),
        "\nSpectrum type: Load from file..",
        "\nSpectrum file: user-defined p-winds spectrum",
        "\nUse only EUV? False",
        "\n[E_low,E_mid,E_high] = [ 13.60 - 123.98 - 1.24e3 ]",
        "\nLog10 of X-ray luminosity [erg/s]: " + str(log_lx),
        "\nLog10 of EUV luminosity [erg/s]: " + str(log_leuv),
        "\nGrid type: Mixed",
        "\nNumerical flux: HLLC",
        "\nReconstruction scheme: PLM",
        "\nRelative momentum threshold: " + str(momentum_threshold),
        "\nInclude He23S? False",
        "\nLoad IC? False",
        "\nDo only PP: False",
        "\nForce start: False"
    ]

    with open(input_file, 'a') as f:
        for in_str in input_strings:
            f.write(in_str)
    f.close()

    # Compile and execute ATES
    run_ates(compiler)

    # Read output files
    output_data_0 = np.loadtxt(_ATES_DIR + "/output/Hydro_ioniz.txt")
    output_data_1 = np.loadtxt(_ATES_DIR + "/output/Ion_species.txt")

    results = {
        'r': output_data_0[:, 0],  # Radial distance in Planetary radii
        'density': output_data_0[:, 1],  # Mass density in unit of proton mass
        'velocity': output_data_0[:, 2],  # Velocity in unit of scale velocity
        'pressure': output_data_0[:, 3],  # In units of cgs
        'temperature': output_data_0[:, 4],  # In unit of K
        'heating_rate': output_data_0[:, 5],  # In units of CGS
        'cooling_rate': output_data_0[:, 6],  # In units of CGS
        # 'heating_efficiency': output_data_0[:, 7],  # Adimensional
        'n_h_i': output_data_1[:, 1],  # Neutral H number density
        'n_h_ii': output_data_1[:, 2],  # Ionized H number density
        'n_he_i': output_data_1[:, 3],  # Neutral He number density
        'n_he_ii': output_data_1[:, 4],  # Singly-ionized He number density
        'n_he_iii': output_data_1[:, 5]  # Doubly-ionized He number density
    }

    return results


# Code snippet that compiles the ATES routines and executes the main loop
def run_ates(compiler='gfortran'):
    """
    Compiles the ATES Code Fortran routines and executes the main loop.

    Parameters
    ----------
    compiler : `str`, optional
        Choose which compiler to use in the compilation of ATES. The available
        options are ``'gfortran'`` and ``'ifort'``. Default is ``'gfortran'``.
    """
    # To honor the original ATES interface, we print some info here
    print('Compiling the ATES Fortran modules using {}.'.format(compiler))

    # Define some important directory locations
    ATES_DIR = _ATES_DIR + "/"
    SRC_DIR = ATES_DIR + "src/"
    DIR_MOD = SRC_DIR + "mod/"
    DIR_FILES = SRC_DIR + "modules/files_IO/"
    DIR_FLUX = SRC_DIR + "modules/flux/"
    DIR_FUNC = SRC_DIR + "modules/functions/"
    DIR_INIT = SRC_DIR + "modules/init/"
    DIR_NLSOLVE = SRC_DIR + "modules/nonlinear_system_solver/"
    DIR_PPC = SRC_DIR + "modules/post_process/"
    DIR_RAD = SRC_DIR + "modules/radiation/"
    DIR_STAT = SRC_DIR + "modules/states/"
    DIR_TIME = SRC_DIR + "modules/time_step/"
    output_dir = ATES_DIR + "output/"

    # Create some directories if necessary
    if os.path.exists(output_dir) is not True:
        os.mkdir(output_dir)
    else:
        pass
    if os.path.exists(DIR_MOD) is not True:
        os.mkdir(DIR_MOD)
    else:
        pass

    # Clean old files
    try:
        os.remove(ATES_DIR + "ATES.x")
    except OSError:
        pass

    for f in glob.glob(DIR_MOD + "*.mod"):
        try:
            os.remove(f)
        except OSError:
            pass

    # Define compiler
    if compiler == 'gfortran':
        first_string = [
            'gfortran',
            '-J' + DIR_MOD,
            '-I' + DIR_MOD,
            '-fopenmp',
            '-fbacktrace'
        ]
    elif compiler == 'ifort':
        first_string = [
            'ifort',
            '-03',
            '-module',
            DIR_MOD,
            '-qopenmp',
            '-no-wrap-margin'
        ]
    else:
        raise ValueError('The only supported compilers are gfortran and ifort.')

    # Compile the necessary modules
    compile_str = first_string + [
        DIR_INIT + "parameters.f90",
        DIR_FILES + "input_read.f90",
        DIR_FILES + "load_IC.f90",
        DIR_FILES + "write_output.f90",
        DIR_FILES + "write_setup_report.f90",
        DIR_FUNC + "grav_field.f90",
        DIR_FUNC + "cross_sec.f90",
        DIR_FUNC + "UW_conversions.f90",
        DIR_FUNC + "utilities.f90",
        DIR_NLSOLVE + "dogleg.f90",
        DIR_NLSOLVE + "enorm.f90",
        DIR_NLSOLVE + "hybrd1.f90",
        DIR_NLSOLVE + "qform.f90",
        DIR_NLSOLVE + "r1mpyq.f90",
        DIR_NLSOLVE + "System_HeH.f90",
        DIR_NLSOLVE + "System_HeH_TR.f90",
        DIR_NLSOLVE + "System_implicit_adv_HeH.f90",
        DIR_NLSOLVE + "System_implicit_adv_HeH_TR.f90",
        DIR_NLSOLVE + "dpmpar.f90",
        DIR_NLSOLVE + "fdjac1.f90",
        DIR_NLSOLVE + "hybrd.f90",
        DIR_NLSOLVE + "qrfac.f90",
        DIR_NLSOLVE + "r1updt.f90",
        DIR_NLSOLVE + "System_H.f90",
        DIR_NLSOLVE + "System_implicit_adv_H.f90",
        DIR_RAD + "sed_read.f90",
        DIR_RAD + "J_inc.f90",
        DIR_RAD + "Cool_coeff.f90",
        DIR_RAD + "util_ion_eq.f90",
        DIR_RAD + "ionization_equilibrium.f90",
        DIR_NLSOLVE + "T_equation.f90",
        DIR_PPC + "post_process_adv.f90",
        DIR_STAT + "Apply_BC.f90",
        DIR_STAT + "PLM_rec.f90",
        DIR_STAT + "Source.f90",
        DIR_STAT + "Reconstruction.f90",
        DIR_FLUX + "speed_estimate_HLLC.f90",
        DIR_FLUX + "speed_estimate_ROE.f90",
        DIR_FLUX + "Num_Fluxes.f90",
        DIR_TIME + "RK_rhs.f90",
        DIR_TIME + "eval_dt.f90",
        DIR_INIT + "define_grid.f90",
        DIR_INIT + "set_energy_vectors.f90",
        DIR_INIT + "set_gravity_grid.f90",
        DIR_INIT + "set_IC.f90",
        DIR_INIT + "init.f90",
        ATES_DIR + "ATES_main.f90",
        "-o",
        ATES_DIR + "ATES.x"
    ]
    subprocess.run(compile_str)

    # And now run ATES
    print('Running the main loop.')
    execute_str = ATES_DIR + "ATES.x"
    subprocess.run(execute_str)
    print('ATES shutdown.')
