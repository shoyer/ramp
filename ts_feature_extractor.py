import numpy as np

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

class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        """Compute the single variable of mean temperatures in the El Nino 3.4
        region."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = range(n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        enso = get_enso_mean(temperatures_xray['tas'])
        enso_valid = enso.values[valid_range]


        year_fraction = (np.array(valid_range) % 12) / 12.0
        seasonal_features = np.vstack(
            [np.vstack([np.sin(2 * np.pi * year_fraction / n),
                        np.cos(2 * np.pi * year_fraction / n)])
             for n in range(1, 3)]).T
        X = np.c_[enso_valid[:, np.newaxis], seasonal_features]
        return X
