#
# Daniel Pflueger, created 2021-11-29
# d.pfluger@uu.nl
#
# Licensed under Creative Commons - Attribution-NonCommercial-ShareAlike 4.0 International
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/
#

import xarray as xr
import numpy as np
import geo_tools as gt
import os

#
# Constants
#

earth_surface_area = 5.1*10**14 # in m^2
g = 9.8 # m/s^2

#
# Normalization functions
#
# Define integrals for fields that can be used as normalization
# Use the format "int_test(ds,name) -> xr.DataArray" when writing a new function
# This format is required by build_expected_field
#

def int_l2_lev(ds,name):
    '''
    Compute the daily L2 norm of a field using level index coordinates:
    norm of field = sqrt(integrate dlat cos(lat) dlevel_index field(level_index,lat))

    Parameters:
    ---
    ds (xr.Dataset): Dataset containing field data
    name (string): Name of the field such that ds[name] returns the field

    Returns:
    ---
    xr.DataArray: Daily normalization constants
    '''

    field = ds[name]

    # as we want to compute the L2 norm of the field, take the square of the field values
    # among other things, this ensures positivity of the final integral
    integrand = field**2

    # convert lev coordinate to lev indices
    # the lev index axis consists only of dimensionless integer numbers
    field = field.assign_coords({"lev": np.arange(0,np.size(field.lev))})

    # weighting of latitudes
    weights = np.cos(np.deg2rad(field.lat))
    integrand = integrand * weights

    # integrate along lev index and latitudinal direction
    integral = integrand.integrate(coord='lat').integrate(coord='lev')
    integral = np.sqrt(integral)

    return integral

def int_l2_barometric(ds,name):
    '''
    Compute daily L2 norm of a field in position by converting sigma-level coordinates
    to approximated(!) dimensionless height parameter from barometric-like formula: z = 7310m * log(1013hPa/lev)

    Parameters:
    ---
    ds (xr.Dataset): Dataset containing field data and auxilliary variable PS.
                     lev coordinate needs to be in hPa units.
    name (string): Name of the field such that ds[name] returns the field

    Returns:
    ---
    xr.DataArray: Daily normalization constants
    '''

    field = ds[name]

    integrand = field**2

    # convert lev coordinates to 'height' parameter
    # This uses the formula given by Marshall, Plumb in 'Atmosphere, Ocean and Climate Dynamics', 2008
    # Eqn. 3.8 z = H ln(p_S/p) where H=7.31km
    # Choose globally homogeneous sea level pressure p_S = 1013 hPa

    z = 7310 * np.log(1013/ds.lev)
    integrand = integrand.assign_coords({"lev": z})

    # weighting of latitudes
    weights = np.cos(np.deg2rad(field.lat))
    integrand = integrand * weights

    # integrate along lev index and latitudinal direction
    # minus sign because of reversed integration order in lev (= approx height) coordinates
    integral = -integrand.integrate(coord='lat').integrate(coord='lev')
    integral = np.sqrt(integral)

    return integral

def int_mass(ds,name):
    '''
    Compute total mass of an aerosol concentration (aerosol mass/air mass) field

    Parameters:
    ---
    ds (xr.Dataset): Dataset containing field data and auxilliary variables. Must contain auxilliary variables PS, hybi, hyai, P0
    name (string): Name of the field such that ds[name] returns the field

    Returns:
    ---
    xr.DataArray: Daily normalization constants, here: aerosol mass in kg
    '''

    # Compute physical pressures from auxilliary variables
    ps = ds.PS * ds.hybi + ds.hyai * ds.P0

    # Assign level index to field
    field = xr.DataArray(dims=['time','lev','lat'],
        coords = {'time': ds.time, 'lat': ds.lat, 'lev': np.arange(0,np.size(ds.lev))},
        data = ds[name].data)

    # Compute physical pressure differences.
    dps = xr.DataArray(dims=['time','lat','lev'],
        coords = {'time': ps.time, 'lat': ps.lat, 'lev': np.arange(0,np.size(ps.ilev)-1)},
        data = ps.diff(dim='ilev'))

    integral = (field * dps).integrate(coord='lev') # integrate in pressure space using dps are p intervals -> gives column burden at latitudes
    integral = gt.global_mean(integral,zonal=True) # global mean of column burden

    integral = integral/g * earth_surface_area # Final total mass in kg

    # Some output formatting useful in plots
    integral.attrs['long_name'] = 'Aerosol Mass'
    integral.attrs['units'] = 'kg'

    return integral

def int_aod(ds,name):
    '''
    Compute global mean of lat, (also called 'L0')

    Parameters:
    ---
    ds (xr.Dataset):
    name (string): Name of field, should be 'AODVISstdn'. Argument only included for compatibility reasons

    Returns:
    ---
    xr.DataArray: Daily normalization constants, here: AOD global mean
    '''

    field = ds[name]

    integral = gt.global_mean(field)

    # Output formatting for plots
    integral.attrs['long_name'] = 'AODVISstdn Global Mean'
    integral.attrs['units'] = 'dim.less'

    return integral

#
# Expected field
#

def build_expected_field(ds,name,tbounds,
                        norm=int_l2_lev,
                        interpolate_results=False):
    '''
    Normalize an input field and build its expected field.

    Parameters:
    ---
    ds (xr.Dataset): Dataset containing field data as well as auxilliary variables (e.g. hyai, P0) used in normalization
    name (string): Name of the field such that ds[name] returns the field
    tbounds (list or tuple of int): Specify years tbounds[1]<=t<=tbounds[2] from which to compute the expected fields.
    norm (optional, function): Normalizing function. Default is int_l2_lev.
    interpolate_results (Boolean, default=False): Apply linear daily interpolation to expected field results (only 'exp_field')

    Returns:
    ---
    dict:   'exp_field': (xr.DataArray) Expected field with integer time dimension
            'n': (xr.DataArray) Annual normalization (obtained by averaging daily normalization factor)
            'nday': (xr.DataArray) Daily normalization
    '''
    # Step 1: Normalization
    # Compute the annual normalization constants of the input field

    nday = norm(ds,name) # obtain daily normalization constants
    n = nday.groupby('time.year').mean() # annual normalization constants
    norm_field = ds[name].groupby('time.year')/n # normalize input field using a constant normalization within each year given by n

    # Step 2: Averaging
    # Average the normalized field in the specified time interval
    tslice = slice(str(tbounds[0]),str(tbounds[1])) # convert time bounds into format readable by xarray
    norm_field = norm_field.sel(time=tslice)
    exp_field = norm_field.groupby('time.dayofyear').mean()
    if interpolate_results:
        exp_field = interpolate_doy(exp_field)
    exp_field.name = name # add identifier for convenience

    # Prepare and return results
    result_dict = {'exp_field': exp_field, 'n': n, 'nday': nday}
    return result_dict

#
# Aux functions
#

def interpolate_doy(da):
    '''
    Interpolate a spatial field (e.g. lev/lat coordinates)
    with integer day of year time coordinates using xr.interp

    Parameters:
    ---
    da (DataArray): Field to interpolate (with dayofyear time axis)

    Returns:
    ---
    DataArray:  Interpolated field with a filled time axis from 1 to 365
    '''

    # xarray provides the 'interp' function for DataArray
    # It can however only interpolate to points _within_ a time interval
    # To get the time points outside the interval limits,
    # loop the dataset and perform an interpolation

    # add 'virtual' data to DataArray by looping the first data point
    doy_first = int(da.dayofyear[0])
    doy_loop = np.arange(doy_first,doy_first+365) # This excludes the last data point
    # Use slice syntax so that the dayofyear coordinate is retained for later concatenation
    da_append = da.sel(dayofyear=slice(1, doy_first))
    da_append = da_append.assign_coords({'dayofyear': [365+doy_first]})
    da_loop = xr.concat([da,da_append],dim='dayofyear')
    # Perform the interpolation on the looped dataset
    da_loop = da_loop.interp(dayofyear = doy_loop)
    # Get the data after dayofyear=365 to wrap it back around to 1,...,doy_first
    da_wrap = da_loop.sel(dayofyear=slice(366,365+doy_first))
    da_wrap = da_wrap.assign_coords({'dayofyear': np.arange(1,doy_first)})
    # Glue the wrapped data to the rest of the data
    # with the spurious data after time 365 cut off
    da_end = da_loop.sel(dayofyear=slice(1,365)) # note that the first data points will be missing
    da_interp = xr.concat([da_wrap,da_end],dim='dayofyear')

    return da_interp

#
# I/O functions
#
# These functions help you save and load expected field data
#

def save_exp(res_dict,path):
    '''
    Saves the output of build_expected_field into a given directory

    Parameters:
    ---
    res_dict (dict): Output dictionary generated by build_expected_field
    path (string): Save path

    Returns:
    ---
    None
    '''

    assert path[-1]=='/', 'Path must end with /'
    assert res_dict['exp_field'].name is not None, 'Exp. field needs a name. Set it via res[\'exp_field\'].name=name'

    name = res_dict['exp_field'].name

    if not os.path.exists(path):
        print(f'Creating parent directory {path}')
        os.mkdir(path)

    # creating sub-path: makes loading easier!
    sub_path = f'{path}{name}/'
    if not os.path.exists(sub_path):
        print(f'Creating sub-directory {sub_path}')
        os.mkdir(sub_path)
    else:
        print(f'Data directory {sub_path} found. Possibly overwriting data.')

    for res,resv in res_dict.items():
        resv.to_netcdf(f'{sub_path}{res}.nc')

def load_exp(path):
    '''
    Load output saved by save_exp in path

    Parameters:
    ---
    path (string): Load path (should equal path-argument of save_exp)

    Returns:
    ---
    (dict of dict): Dictionary with keywords obtained from names.
                    Items are result dictionaries  similar to the output of build_expected_field
    '''

    assert path[-1]=='/', 'Path must end with /'
    assert os.path.exists(path), f'Path {path} does not exist.'

    names = os.listdir(path)
    results = {}
    for name in names:

        res_dict = {'exp_field': None, 'n': None, 'nday': None}
        for res in res_dict:
            # Open_dataset returns a Dataset. Extract a DataArray from this Dataset
            ds  = xr.open_dataset(f'{path}{name}/{res}.nc')
            var_name = list(ds.keys())[0]
            da = ds[var_name]
            da.name = f'{name}_{res}'
            res_dict[res] = da
        results[name] = res_dict

    return results
