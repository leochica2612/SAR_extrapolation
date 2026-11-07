import numpy as np
import xarray as xr

def check_tile_for_wind_retrieve(tile: xr.Dataset, var_vv:str, var_vh:str, lat:str , lon:str):
    
    sigmas = []
    sigma_vv = tile[var_vv]
    sigma_vh = tile[var_vh]
    sigmas.append(sigma_vv)
    sigmas.append(sigma_vh)
    
    for sigma in sigmas:
        # if "y" not in sigma.coords or "x" not in sigma.coords:
        #     return None, "Not y and x in sigma.coords", None, None, None, None
        if sigma.ndim != 2:
            return None, "More or less than 2 dimensions", None, None, None, None
        
    # SAR geographical coordinates
    # Retrieve lat and lon for each SAR pixel a put in an array
    lats = tile[lat]
    lons = tile[lon]
    lat_array = np.asarray(lats.values, dtype=np.float64)
    lon_array = np.asarray(lons.values, dtype=np.float64)
    
    for sigma in sigmas:
        if lat_array.shape != sigma.shape or lon_array.shape != sigma.shape:
            return None, "Differences between sigmas and lat/lon", None, None, None, None
        
    # Get incidence angle at the same position
    theta_da = tile['incident_angle']
    theta_tp = np.asarray(theta_da.values, dtype=np.float64)
    if theta_tp.ndim != 2:
        return None, "Theta with more or less than 2 dimensions", None, None, None, None
    
    # 3. Platform heading (satellite fligh heading)
    heading_raw = tile['platform_heading'].values

    try:
        platform_heading = float(np.asarray(heading_raw).item())
    except (TypeError, ValueError):
        raise ValueError(f"Could not convert platform_heading={heading_raw!r} to float.")
    
    return sigma_vv, sigma_vh, lat_array, lon_array, theta_tp, platform_heading

def windows_for_tile(level:int, x_idx:int, y_idx:int, nx, ny):
    if level < 1:
        return None, None
    
    y_min = y_idx - level
    y_max = y_idx + level + 1
    x_min = x_idx - level
    x_max = x_idx + level + 1
    y_min = max(0, y_min)
    y_max = min(ny, y_max)
    x_min = max(0, x_min)
    x_max = min(nx, x_max)
    
    return (y_min, x_min), (y_max, x_max)

def dist_and_closest_pixel(lat_array, lon_array, lat0, lon0):
    R = 6_371_000.0  # Earth radius in meters
    # Find closest pixel
    lat0_rad = np.deg2rad(lat0)
    dx = R * np.cos(lat0_rad) * np.deg2rad(lon_array - lon0)
    dy = R * np.deg2rad(lat_array - lat0)
    distance2_m = dx**2 + dy**2
    invalid = ~np.isfinite(lat_array) | ~np.isfinite(lon_array)
    distance2_m[invalid] = np.inf
    y_idx, x_idx = np.unravel_index(np.argmin(distance2_m), distance2_m.shape)
    
    return y_idx, x_idx, distance2_m

def values_windows(array, minimums, maximums):
    return np.asarray(array[minimums[0]:maximums[0], minimums[1]:maximums[1]],dtype=np.float64)

def return_weight(weighting, sigma_vv, sigma_vh, distance2):
    valid = (np.isfinite(distance2) & np.isfinite(sigma_vv))
    if weighting is None:
        # No weighting: ordinary least squares
        weights_vv_window = np.ones_like(sigma_vv, dtype=np.float64)
        weights_vh_window = np.ones_like(sigma_vh, dtype=np.float64)
        
    elif weighting == "uniform":
        # Flat distribution, normalized to sum to 1
        weights_vv_window = valid.astype(np.float64)
        weight_sum = weights_vv_window.sum()
        if weight_sum == 0: return None
        weights_vv_window /= weight_sum
        
    elif weighting == "inverse_distance2":
        weights_vv_window = np.zeros_like(sigma_vv, dtype=np.float64)
        weights_vv_window[valid] = 1.0 / np.maximum(distance2[valid], 1.0)
        
    elif weighting == "sigma":
        weights_vv_window = 1/sigma_vv
        
    else:
        return None
    
    return np.sqrt(weights_vv_window)

def return_bounds(z_ref: float = 60.0, scaled: bool = False):
        low_level = 0.01
        intermediate_level = 500.0
        upper_level = 3000.0
        
        if scaled:
            fits = [
                ("u", (z_ref/-low_level, z_ref/-intermediate_level)),
                ("nu", (z_ref/-intermediate_level, z_ref/-upper_level)),
                ("ns", (z_ref/upper_level, z_ref/intermediate_level)),
                ("s", (z_ref/intermediate_level, z_ref/low_level))
            ]
        else:
            fits = [
                ("u", (-intermediate_level, -low_level)),
                ("nu", (-upper_level, -intermediate_level)),
                ("ns", (intermediate_level, upper_level)),
                ("s", (low_level, intermediate_level)),
            ]
        return fits