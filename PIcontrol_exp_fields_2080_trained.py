# Explicit feedback code suite
# Sample control parameters file
#
# Written by Ben Kravitz (ben.kravitz@pnnl.gov or ben.kravitz.work@gmail.com)
# Last updated 29 November 2016 -> note D.P.: see comment below
#
# This script provides information about the feedback algorithm.  All of this
# is user-customizable.  The other parts of the script will give you outvals,
# lats, lons, and times.  The output of this script should be a list called
# nlvals, which consists of pairs.  The first item in the pair is the name
# of the namelist value.  The second item is the value itself as a string.
#
# This script is written in native python format.  Be careful with brackets [],
# white space, and making sure everything that needs to be a string is actually
# a string by putting it in quotes ''.  All lines beginning with # are comments.

#
# Modified by Daniel Pflueger, Claudia Wieners and Leo Kampenhout on 2021-10-01
# for use in IMAU CESM-CAM 6 SAI in RCP 8.5 scenario with PI+1.5 degC target run
# Contacts: d.pfluger@uu.nl, C.E.Wieners@uu.nl, l.vankampenhout@uu.nl
#

#
# Comment 2022-02-24: This version of the feedback controller dynamically introduces the integrator depending on two conditions:
# The temperature error terms are summed after...
# - the temperature error is within a defined distance of the reference temperature
# OR
# - a defined amount of years have past since the initialization of the controller
# After any of these conditions is met, the integrator remains switched on indefinitely
#

#
# Comment 2022-06-27: The feedforward used in this controller file is trained on the SAI 2020 run:
# the last 30 years of AOD in the SAI2020 run are linearly interpolated and put into SAI 2080 as a feedforward
#

from __future__ import print_function
import os
import strataero_forcing
import norm_const_fit as nc
import xarray as xr

#### IMPORTANT FILE PATHS ####

fit_consts_path = os.path.join(maindir,'fit_params_2022_02_09/') # path to fit parameters used by nc.all_norms_from_main_field
exp_fields_path = os.path.join(maindir,'exp_fields_2022_02_09/') # path to exp fields used by strataero_forcing.load_exp_fields

# strataero file
# this is the CURRENT FILE, i.e. that is continuously updated
# and referenced in the user namelist: user_nl_cam

#prescribed_strataero_datapath		= '/projects/0/uuesm/GeoEng/feedback/'
prescribed_strataero_datapath		= f'/projects/0/uuesm2/archive/{casename}/strataero'
prescribed_strataero_file		= 'ozone_strataero_WACCM_L70_zm5day_2015-2100_SSP585_CAMfeedback.nc'

if (not os.path.exists(prescribed_strataero_datapath)):
    print('INFO: creating '+prescribed_strataero_datapath)
    os.makedirs(prescribed_strataero_datapath)
    assert(os.path.exists(prescribed_strataero_datapath))

strataero_path_namelist = os.path.join(prescribed_strataero_datapath ,prescribed_strataero_file)


# This is the original strataero, only needed once (at the start)
strataero_path_cnt = '/projects/0/uuesm/inputdata/atm/cam/ozone_strataero/ozone_strataero_WACCM_L70_zm5day_2015-2100_SSP585_c190529.nc'


#### PREPARE LOG ####

logfilename='ControlLog_'+casename+'.txt'

logheader=['Timestamp','dT0','sum(dT0)','L0','int_weight',
           'n_AODVISstdn',
           'n_so4mass_a1','n_so4mass_a2','n_so4mass_a3',
           'n_diamwet_a1','n_diamwet_a2','n_diamwet_a3',
           'n_SAD_AERO']

# 'firsttime' is a flag value that is set to 1 if the script is run for the first time
# in this first run, an output log is created
firsttime=0
if os.path.exists(maindir+'/'+logfilename)==False: # where is maindir defined?
    firsttime=1
    print('INFO: first time')

else:
    loglines=readlog(maindir+'/'+logfilename)
    print('INFO: continuation')


#### USER-SPECIFIED CONTROL PARAMETERS ####

# Note: feedforward slope and baseyear were changed using SAI 2020 output

refvals=[288.73] # 288.78 would be 1.5 deg above the last 100yr of PI
kivals=[0.028] # integration parameter of feedback "k_i"
kpvals=[0.028] # proportional parameter of feedback "k_p"
kff = 0.0096# feedforward constant
firstyear=2080 # starting year of feedback
baseyear=2028  # only to define the dt for computing the feedforward. 1.5K threshold crossed at ~2028
# Integrator ramp-up: summation of error terms is ramped up over a time interval
# This is achieved by multiplying error terms with a weight factor from 0 to 1 that is linearly raised over a given time interval
int_onset_threshold = 0.5 # Temperature deviation threshold for onset of integrator
int_failsafe_time = 6 # Time interval after which to start integrator in case other onset condition is not met

#### FEEDBACK COMPUTATION ####

# compute the temperature goals (in our case, only T0 which is global mean surface temp.) and error terms

w=makeweights(lats,lons)
T0=numpy.mean(gmean(outvals[0],w))
de=numpy.array([T0-refvals[0]]) # error terms: last years' deviation from the goal

# Extract previous timestamp from log (or initialize if this is the first time)
if firsttime==1:          #the sum over all previous errors (integral term)
    timestamp=firstyear
else:
    timestamp=int(loglines[-1][0])+1
# To prevent mistakes when reading the log file (e.g. indexation errors), test the obtained timestamp
assert timestamp>1900 and timestamp<3000, 'Unrealistic timestamp from previous year. Are you reading the log file correctly?'

int_weight = 0 # Integrator weight is binary: 0 or 1
if not firsttime==1: # Once integrator has been activated, always leave it on / Can only be executed if log file already exists, hence 'not firsttime' is required
    previous_int_weight = int(loglines[-1][4])
    # Check that previous int_weight was read correctly
    assert previous_int_weight==0 or previous_int_weight==1, 'Non-binary int weight from previous year. Are you reading the log file correctly?'
    if previous_int_weight==1:
        int_weight = 1
if abs(de)<int_onset_threshold: # Activate integrator once temperature error is within acceptable range
    int_weight = 1
if timestamp >= (firstyear + int_failsafe_time): # Activate integrator after defined time period has passed
    int_weight = 1

if firsttime==1:          # Computing sum of error terms for integrator. This includes the binary weight factor int_weight
    sumde=de*int_weight
else:
    sumdt0=float(loglines[-1][2])+(T0-refvals[0])*int_weight # include weight factor for error sum/this is in implementation of the integrator ramp-up
    sumde=numpy.array([sumdt0])

dt = timestamp - baseyear
print(f'INFO: year = {timestamp}')


#first, we compute the feedforward and feedback, hence the total "geoengineering strength", in terms of AOD patterns (only l0 in our case)

# feedforward
# based on the WACCM simulation, but scaled with CAM_WACCM_corrfac to take into account that CAM needs more cooling.
l0hat=kff*dt

# feedback
l0kp1=kpvals[0]*de[0]+kivals[0]*sumde[0]

# all of the feeds
l0step4=l0kp1+l0hat

l0=max(l0step4,0)
ell=numpy.array([[l0]])


### UPDATING FORCING FILE ###


if (firsttime):
    print(f'INFO: base file: {strataero_path_cnt}')
    strataero_path = strataero_path_cnt
else:
    # archive current strataero file, adding the year as a label
    label = timestamp-1
    new_path = strataero_path_namelist.replace('.nc', f'-{label}.nc')
    if (os.path.isfile(new_path)):
        print(f'INFO: current strataero file has already been archived? Skipping archive step')
        if (os.path.isfile(strataero_path_namelist)):
            raise RuntimeError(f'current strataero file {strataero_path_namelist} needs to be removed manually')
    else:
        assert(os.path.isfile(strataero_path_namelist) == True)
        print(f'INFO: archiving current strataero file as {new_path}')
        os.rename(strataero_path_namelist, new_path)
    print(f'INFO: base file: {new_path}')
    strataero_path = new_path

print(f'INFO: storing result in {strataero_path_namelist}')

# Load expected fields
exp_fields = strataero_forcing.load_expected_fields(exp_fields_path )

with xr.open_dataset(strataero_path) as strataero_ds:

    aod_norm = l0 

    norm_consts = nc.all_norms_from_main_field(aod_norm,fit_consts_path)
    norm_consts['AODVISstdn'] = aod_norm

    print('INFO: calling strataero_forcing.update_strataero()')
    strataero_forcing.update_strataero(strataero_ds,timestamp,norm_consts,exp_fields) # IN-PLACE update of strataero_ds
    print('INFO: leaving strataero_forcing.update_strataero()')


    # Note LvK: need to write NetCDF3, as NetCDF4 does seem to give errors within CESM
    strataero_ds.to_netcdf(strataero_path_namelist, format='NETCDF3_64BIT')


#### WRITE LOG ####

newline = [str(timestamp),str(de[0]),str(sumde[0]),str(l0),str(int_weight),
           str(norm_consts['AODVISstdn']),
           str(norm_consts['so4mass_a1']),str(norm_consts['so4mass_a2']),str(norm_consts['so4mass_a3']),
           str(norm_consts['diamwet_a1']),str(norm_consts['diamwet_a2']),str(norm_consts['diamwet_a3']),
           str(norm_consts['SAD_AERO'])]

if firsttime==1:
    linestowrite=[logheader,newline]
else:
    linestowrite=[]
    for k in range(len(loglines)):
        linestowrite.append(loglines[k])
    linestowrite.append(newline)

writelog(maindir+'/'+logfilename,linestowrite)
