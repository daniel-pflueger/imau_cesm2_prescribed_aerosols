#! /usr/bin/env python
# Explicit feedback code suite
# Main setup script
#
# Written by Ben Kravitz (bkravitz@iu.edu or ben.kravitz.work@gmail.com)
# Last updated 11 July 2019
#
# This script provides the main information about the run.  Note that everything
# here is specific to an individual run.  Each run should have its own copy
# of the feedback suite.
#
# This script is written in native python format.  Be careful with brackets [],
# white space, and making sure everything that needs to be a string is actually
# a string by putting it in quotes ''.  All lines beginning with # are comments.
#
# Things you need to specify:
#
################## GLOBAL SETUP PARAMETERS ##################
#
#   casepath = The path where this specific case is built and run.
#
#   maindir = The path where the control suite sits.
#
#   scratchdir = The script will need to extract and process some of the model
#                output.  This is the path where it will do that.  The scratchdir
#                will be removed after the script is done processing everything,
#                so make sure this does not point to a directory that you care about.
#
#   runname = The name of the case you're running.
#
#   frequency = How often you want the controller to operate.  This
#              includes two pieces of information:  a number and a letter.
#              The letters can be d (days), w (weeks), m (months),
#              or y (years).  For example, if you want the controller to run
#              every 23 days, you would put '23d' (the quotes are important
#              because this is a string).  You're probably going to mostly use
#              '1m' or '1y'.  There is currently no capability of having sub-daily
#              frequency.  If the frequency of the model output is not concordant
#              with what you select here (e.g., you only output monthly, but you
#              select frequency='2w'), you will get unpredictable behavior.
#
#   maxrest = The number of times you want the controller to restart.
#              For example, if you want a 50 year simulation, and if
#              frequency='1y', then maxrest=49.
#
#   pathtocontrol = The absolute path of the feedback algorithm.
#
################## VARIABLE-SPECIFIC SETUP PARAMETERS ##################
# These are lists that are specific to each variable.
#
#   variables = The variables that the feedback algorithm needs.  Note that
#     for now, all variables must be 2-D fields (i.e., no vertical component).
#
#   archivepaths = The paths where all of your archived output is stored, one
#     for each variable.  This is split apart in case you want to call for inputs
#     from two different model components (e.g., atmosphere and ocean).
#     
#   varlocs = The locations of the variables to be extracted.  Examples include
#     'cam.h0', 'cam.h2', or 'pop.h.nday1'.

# Control simulation
# /home/kampe004/uuesm2/archive/b.e21.BSSP585cmip6.f09_g17.control.01/atm/hist
import os

casename='b.e21.BSSP585cmip6.f09_g17.feedback.08' 
print(f'INFO: casename = {casename}')

casepath='/home/kampe004/cesm/my_cases/cesm2_geo/'+casename
scratchdir='/scratch-local/kampe004/'+casename+'/temp_stuff' 
frequency='1y'
maxrest=99

maindir=os.path.join(os.getcwd(),'controller') # assume we execute from CASEROOT DIR
pathtocontrol=maindir+'/PIcontrol_exp_fields.py'

variables=['TREFHT']
varlocs=len(variables)*['cam.h0']


DO_BRANCH=False # set to True when running for first time , and branching from other run

# LvK 2022.02.21 existence of log file as test for first time
logfilename='ControlLog_'+casename+'.txt'
if os.path.exists(maindir+'/'+logfilename)==False: 
    DO_BRANCH=True

if (DO_BRANCH): 
    # read temperature from reference run
    runname = 'b.e21.BSSP585cmip6.f09_g17.control.01' 
    branch_year = 2080
    hist_dir = f'/projects/0/uuesm2/archive/{runname}_{branch_year}/atm/hist'    
    
    print('=========='*10)
    print(f'WARNING: branching')
    print(f'WARNING: reading history files not from {casename}, but from {runname}')
    print('=========='*10)

    # LvK: the driver.py script is so simple, it will always read the last in the archive folder
    # this is not always what you want, e.g. when the branch sim ran until 2100 but you wish to branch
    # in an earlier year, say 2060. 
    # Easiest work-around is to create a separate folder with symlinks to just a single year
    # this folder is named 'runname_YEAR' to indicate the branch year YEAR
    archivepaths=len(variables)*[hist_dir]

else:
    # read temperature from current case (continuation)
    runname = casename
    archivepaths=len(variables)*['/projects/0/uuesm2/archive/'+runname+'/atm/hist'] # do NOT include a '/' at the end of each path'
    print('INFO: continuation')


print('INFO: reading h0 files from sim '+runname)

# choose one of: 'CAM', 'WACCM'
atm_comp='CAM'

###########################
### CALLING MAIN SCRIPT ###
###########################

assert(atm_comp in ('CAM', 'WACCM'))


#execfile(maindir+"/driver.py")
print('INFO: calling driver.py')
exec(open(maindir+"/driver.py").read()) # LvK: Py3

# runs preview_namelists 
#os.system(casepath+'/preview_namelists')
