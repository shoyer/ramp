import numpy as np
import xray

import urllib
import os.path
from sklearn.decomposition import PCA

# download the sea mask
if not os.path.exists('mask.nc'):
    urllib.urlretrieve(
        'https://drive.google.com/uc?export=download&id=0B-CxJMRyTT32el85MFRSYnU5TGM',
        'mask.nc')

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120

def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(dim=('lat','lon'))

def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region at all time points."""
    return get_area_mean(tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)


def get_sea_mask(ds):
    raw_mask = xray.open_dataset('mask.nc')
    # extract places where the nearest latitude or longitude (before or after)
    # is in the ocean
    sea_mask = ((raw_mask.reindex_like(ds, method='pad').sftlf < 100)
                & (raw_mask.reindex_like(ds, method='backfill').sftlf < 100))
    return sea_mask


def apply_sea_mask(feature_matrix, ds):
    sea_mask = get_sea_mask(ds)
    sea_columns = sea_mask.values.flatten()
    return feature_matrix[:, sea_columns]


def feature_all_world(temperatures_xray, n_burn_in, n_lookahead, skf_is):
    """Use world temps as features."""
    # Set all temps on world map as features
    all_temps = temperatures_xray['tas'].values
    time_steps, lats, lons = all_temps.shape
    all_temps = all_temps.reshape((time_steps, lats * lons))
    all_temps = all_temps[n_burn_in:-n_lookahead, :]
    return all_temps


def feature_seasonal(temperatures_xray, n_burn_in, n_lookahead, skf_is):
    year_fraction = (np.array(valid_range) % 12) / 12.0
    seasonal_features = np.vstack(
        [np.vstack([np.sin(2 * np.pi * year_fraction / n),
                    np.cos(2 * np.pi * year_fraction / n)])
         for n in range(1, 3)]).T
    return seasonal_features


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        """Compute the single variable of mean temperatures in the El Nino 3.4
        region."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_time_index = slice(n_burn_in, -n_lookahead)
        temperatures_xray = temperatures_xray.isel(time=valid_time_index)

        # enso = get_enso_mean(temperatures_xray['tas'])
        # enso_valid = enso.values[valid_range, np.newaxis]

        tropical_ds = temperatures_xray.sel(lat=slice(-30, 30))
        time_steps = temperatures_xray.dims['time']
        tropical_temps = tropical_ds.tas.values.reshape((time_steps, -1))

        sea_temps = apply_sea_mask(tropical_temps, tropical_ds)

        return temperatures_xray.tas.values.reshape((time_steps, -1))

        # return tropical_temps
        # return sea_temps

        # pca_temps = PCA(n_components=10).fit_transform(sea_temps)

        # seasonal_features = feature_seasonal(temperatures_xray, n_burn_in, n_lookahead, skf_is)

        # X = np.c_[pca_temps, seasonal_features]
        # return pca_temps
