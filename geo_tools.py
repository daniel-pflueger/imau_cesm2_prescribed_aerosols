#
# Daniel Pflueger, created on 2021-29-11 
# d.pfluger@uu.nl
#


import xarray as xr
import numpy as np

def global_mean(da,**kwargs):
    """
    Takes an xarray DataArray defined over a lat(+lon optionally) field and returns global mean computed with correct weighting factors.

    Parameters:
    ---
    da (xr.DataArray): DataArray containing a field defined over latitudes (lat) and longitudes (lon) and optionally time
    kwargs (optional): Kept for backwards compatibility ('zonal' keyword-argument used in old versions).

    Returns:
    ---
    float or DataArray: Global mean of the field (over time)
    """

    # Check if DataArray is already zonally integrated
    zonal = True
    if 'lon' in list(da.dims):
        zonal = False # 'lon' dimensions found -> da is not zonally integrated

    # Determine the grid size
    # Note: this requires an equally spaced coordinate grid
    dlat = np.deg2rad(np.diff(da.lat.data)[0])

    # zonal integration
    if not zonal:
        dlon = np.deg2rad(np.diff(da.lon.data)[0])
        da_zonal = 1.0/(2.0*np.pi) * da.sum(dim="lon") * dlon
    else:
        da_zonal = da

    # meridional integration
    weight = 0.5 * np.cos(np.radians(da.lat)) * dlat # diff. element for meridional int. (higher latitudes give lower contribution)
    da_weighted = da_zonal * weight

    return da_weighted.sum(dim="lat")
