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
from warnings import warn

# Find $ATES_DIR environment variable
try:
    _ATES_DIR = os.environ["ATES_DIR"]
except KeyError:
    _ATES_DIR = None
    warn("Environment variable ATES_DIR is not set.")


def ates_model():
    pass


def run_ates():
    # Define some important directory locations
    ATES_DIR = _ATES_DIR + "/"
    SRC_DIR = ATES_DIR + "src/"
    UTILS_DIR = ATES_DIR + "utils/"
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

    # Compile the necessary modules
    compile_str = [
        'gfortran',
        '-J' + DIR_MOD,
        '-I' + DIR_MOD,
        '-fopenmp',
        '-fbacktrace',
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
    execute_str = ATES_DIR + "ATES.x"
    subprocess.run(execute_str)


def make_input_file():
    pass
