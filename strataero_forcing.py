#
# Daniel Pflueger, created 2022-01-03
# d.pfluger@uu.nl
#


import expected_fields as xf
import xarray as xr
import numpy as np

def update_strataero(strataero_ds,year,norm_consts,exp_fields,extra_day=True):

    '''
    Wrapper for computing scaled expected fields using norm constants and updating data variables strataero_ds

    Parameters:
    ---
    strataero_ds (xr.Dataset): strataero Dataset as required by CESM2-CAM6
    year (int): year which to update in strataero_ds
    norm_consts (dict str:float): dictionary of norm constants to multiply with expected fields. keys must be data variable names in strataero_ds
    exp_fields (dict str:xr.Dataset): dictionary of expected field Datasets, keys of dictionary must agree with norm_consts
    extra_day (boolean): Also modify the first day of 'year+1' by repeating the last data in 'year'. This avoids jumps due to interpolation errors in CESM2.

    Returns:
    ---
    function is in-place and modifies strataero_ds
    '''

    scaled_fields = _compute_summands(norm_consts,exp_fields)

    #for field_name in exp_fields:
    for field_name in scaled_fields.keys():
        summand = scaled_fields[field_name]
        _update_variable_in_strataero(strataero_ds,year,field_name,summand,extra_day=extra_day)

    # add a note to strataero_ds that this year has been updated
    # you can use this as a check before executing update_strataero
    # in case the feedback script gets called multiple times in a simulated year
    if not ('feedback_hist' in strataero_ds.attrs.keys()):
        strataero_ds.attrs['feedback_hist'] = str(year)
    else:
        strataero_ds.attrs['feedback_hist'] = strataero_ds.attrs['feedback_hist'] + ' ' + str(year)

def _update_variable_in_strataero(strataero_ds,year,variable_name,summand,extra_day):

    '''
    Updates a specific year's data of a given data variable in a strataero forcing dataset.
    This is done by adding a term to the data variable.

    Parameters:
    ---
    strataero_ds (xr.Dataset): strataero Dataset as required by CESM2-CAM6
    year (int): year in which to modify data
    variable_name (string): name of data variable which to modify, e.g. 'so4mass_a3'
    summand (xr.DataArray): DataArray with integer day of year time axis (in a no-leap year).
                          Should contain a data variable compatible with the variable being updated.
                          Must only have a single data variable
                          Summand is typically obtained from scaled expected fields data

    Returns:
    ---
    function is in-place and modifies strataero_ds
    '''

    # convert doy axis of summand to cftime
    year = str(year)
    summand_cftime = xr.cftime_range(start=year+'-01-01',end=year+'-12-31',calendar='noleap')
    summand = summand.rename({'dayofyear': 'time'})
    summand['time'] = summand_cftime

    assert summand.get_axis_num('time')==0, 'Time must be 0th dimension of field' # Important for some data access application

    # now weed out the superfluous time steps by masking with the time steps present in strataero_ds
    time_mask = strataero_ds.sel(time=year).time
    # look for nearest match along time axis,
    # because summand and strataero time coordinates might have different times of day,
    # e.g. 12:00 and 0:00 o'clock
    summand = summand.sel(time=time_mask,method='nearest')
    # Replace the time axis by the time mask. This sets the times of day to be equal
    summand['time'] = time_mask

    # change name of data variable in summand to be consistent with the field we want to change
    # this is necessary to use the + operator of xarray
    # current_name = list(summand.data_vars)[0]
    # summand = summand.rename({current_name: variable_name})

    # numpy array of updated data
    updated_data = (strataero_ds[variable_name] + summand).data

    # insert updated data into strataero_ds
    # this modification works in-place
    if extra_day:
        # if the extra day for interpolation is included, do not change the first entry in the year
        # it was already updated in the previous iteration when using the interpolation
        if 'feedback_hist' in strataero_ds.attrs.keys(): # the first day is only filtered out if this is not the first time the forcing file has been updated
            strataero_ds[variable_name].loc[dict(time=time_mask[1:])] = np.clip(updated_data[1:,:],0.0,None)
        else: # first iteration: do not omit first day in the year
            strataero_ds[variable_name].loc[dict(time=time_mask)] = np.clip(updated_data,0.0,None)
    else:
        strataero_ds[variable_name].loc[dict(time=time_mask)] = np.clip(updated_data,0.0,None)

    # update the first day of the _following_ year
    # this reduces the risk of interpolation artefacts
    if extra_day:
        extra_day_data = updated_data[-1,:] # last day of the year is repeated for first day of next year
        first_day = strataero_ds.sel(time=str(int(year)+1)).time[0]
        strataero_ds[variable_name].loc[dict(time=first_day)] = extra_day_data
        # for debugging purposes: add date until which we interpolate to dataset attributed
        if not ('interpolated_until' in strataero_ds.attrs.keys()):
            strataero_ds.attrs['interpolated_until'] = str(first_day.data)
        else:
            strataero_ds.attrs['interpolated_until'] = strataero_ds.attrs['interpolated_until'] + ' ' + str(first_day.data)

def _compute_summands(norm_consts,exp_fields):

    '''
    Compute summands to be later used in update_strataero by multiplying expected fields with norm factors

    Parameters:
    ---
    norm_consts (dict str:float): dictionary of norm constants to multiply with expected fields
    exp_fields (dict str:xr.Dataset): dictionary of expected field Datasets, keys of dictionary must agree with norm_consts

    Returns:
    ---
    scaled_fields (dict str:xr.Dataset): dictionary of scaled fields (exp field * norm const). keys are original field names
    '''

    scaled_fields = {}

    # iterate over keys in exp_fields dictionary
    # for each of the fields exp_fields[field_name], the associated norm factor is multiplied to the field
    for field_name in exp_fields:
        scaled_fields[field_name] = exp_fields[field_name] * norm_consts[field_name]

    return scaled_fields

# Functions for extending strataero forcing file

def _create_ts(years):
    ts_container = []
    for year in years:
        ts_append = xr.cftime_range(start=f'{year}-01-01',end=f'{year}-12-31',freq='5D',calendar='noleap')
        ts_container.append(ts_append)

    ts_final = ts_container[0]
    for ts in ts_container[1:]:
        ts_final = ts_final.append(ts)

    return ts_final

def _da_loop_years(da,periods=1,ts_loop=[2090,2099]):
    da_loop = da.sel(time=slice(str(ts_loop[0]),str(ts_loop[1])))
    da_append_container = []
    for i in range(0,periods):
        ts = np.arange(ts_loop[1]+1+10*(i),ts_loop[1]+10*(i+1)+1)
        da_append = da_loop.copy()
        da_append['time'] = _create_ts(ts)
        da_append_container.append(da_append)
    return xr.concat(da_append_container,dim='time')

def extend_ds(ds,fields,periods=1,ts_bounds=[2090,2099],drop_other_fields=True):
    '''
    Extend fields in a dataset by looping a certain period over and over again

    ds: Dataset
    fields: List of field names, e.g. ['AODVISstdn', 'so4mass_a3']. These fields will be extended
    periods: Number of repetitions
    ts_bounds: List of tim e boundaries (year) that encompass the looped time interval
    drop_other_fields (Boolean): Remove spurious fields from dataset (all those not mentioned in 'fields')
    ts_cnt_max (int, default=2099): last year to include in the original control file. Necessary to avoid merging conflicts

    Returns:
    Dataset with extended variables
    '''

    for field in fields:
        print(f'DEBUG field={field}')
        da = ds[field].copy()
        da_append = _da_loop_years(da,periods=periods,ts_loop=ts_bounds)

        # Create dataset out of extension so that we can use xr.merge
        ds_append = da_append.to_dataset()
        ds = xr.merge([ds,ds_append])

    if drop_other_fields:

        all_fields = list(ds.keys())
        keep_fields = fields
        drop_fields = [field for field in all_fields if not field in keep_fields]
        ds = ds.drop(drop_fields)

    return ds

# I/O operations

def load_expected_fields(load_path):

    '''
    Wrapper for loading expected fields. This is based on 'expected_fields.load_exp'. For reasons of backwards compatibility
    'load_expected_fields' is still kept.

    Parameters:
    ---
    load_path (str): path relative to working directory containing expected fields.
                     don't forget the final slash, e.g. './expected_fields/'

    Returns:
    ---
    dict str:xr.Dataset: dict of all exp fields, keys being their respective names as in forcing file e.g. 'so4mass_a3'
    '''

    results = xf.load_exp(load_path)

    # extract only the expected field results
    results_exp = {}
    for name in results:
        results_exp[name] = results[name]['exp_field']

    return results_exp
