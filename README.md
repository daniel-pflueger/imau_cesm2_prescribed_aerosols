# Readme

Scripts used to prescribe stratospheric aerosols in CESM2-CAM6 based on CESM2-WACCM output. Used at [IMAU](https://www.uu.nl/onderzoek/imau) to perform simulations of feedback-controlled stratospheric aerosol injection. Contributors: Daniel Pflüger, Leo van Kampenhout, Jasper de Jong, Claudia Wieners

The controller component was originally built by [Ben Kravitz](https://github.com/bkravitz/feedback_suite) and updated by Daniele Visioni. See [Kravitz et al. (2016), _Earth Systen Dynamics_](https://esd.copernicus.org/articles/7/469/2016/) for more context.

## Building prescribed fields
Note: to run these scripts, you will need stratospheric aerosol data from CESM2-WACCM. In our case, we use results from [Tilmes et al (2020)](https://doi.org/10.5194/esd-11-579-2020), specically a _SSP5 8.5_ and _Geo SSP5-8.5 1.5_ run. For sake of completeness, we uploaded our final averaged fields (which are later scaled by the controller), see 'Output' below.

* `exp_field_and_norms.ipynb`: notebook in which averaged fields are created
* `example.ipynb`: further comments on computing averaged fields
* `expected_fields.py`: normalization schemes and averaging
* `strataero_forcing.py`: compute scaled fields from average field output and user-specified AOD
* `norm_const_fit.py`: fit functions for field amplitudes

## Output
* `./exp_fields_2022_02_09`, `./fit_params_2022_02_09`: 

## Feedforward-feedback controller:
* `main.py`, `driver.py`, `commonroutines.py`: utilities for running the controller and coupling it to CESM2, written by Ben Kravitz with minor modifications from our side
* `PIcontrol_exp_fields_2020.py`: controller for SAI 2020, adapted from Ben Kravitz' original script
* `PIcontrol_exp_fields_2080.py`: controller for SAI 2080, adapted from Ben Kravitz' original script
* `PIcontrol_exp_fields_trained_int_reset.py`: controller for SAI 2080 (mod), adapted from Ben Kravitz' original script with help of Jasper de Jong