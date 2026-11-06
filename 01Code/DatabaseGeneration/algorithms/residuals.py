from .wind_profile import diabatic_wind_profile, delta_diabatic_wind_profile
from .cmod5_n import calc_sigma0_cmod5_n, calc_sigma0_s1w_nr
import numpy as np

def stability_residuals(params,
    heights, wind_speeds, weights, stability,
    delta_v, scaled, z_ref, heights_combination=None):

    if not delta_v and not scaled:
        u_star, L = params
        wind_pred = diabatic_wind_profile(heights, u_star, L, stability)

    elif not delta_v and scaled:
        u_star, eta = params
        wind_pred = diabatic_wind_profile(heights, u_star, eta, z_ref=z_ref, scaled=scaled, stability=stability)

    elif delta_v and not scaled:
        u_star, L = params
        wind_pred = delta_diabatic_wind_profile(z2=heights_combination[:, 0], z1=heights_combination[:, 1],
            u_star=u_star, obu_parameter=L, stability=stability)

    elif delta_v and scaled:
        u_star, eta = params
        wind_pred = delta_diabatic_wind_profile(z2=heights_combination[:, 0], z1=heights_combination[:, 1],
            u_star=u_star, obu_parameter=eta, z_ref=z_ref, scaled=scaled, stability=stability)

    res = np.sqrt(weights) * (wind_pred - wind_speeds)

    return res

def scatter_residuals(params,
    wdir, look_azimuth, theta, sigma0_vv, sigma0_vh, sqrt_weights):
    
    v = float(params[0])
    wind_to = (wdir + 180.0) % 360.0 # Normalize direction.

    phi_rel = (wind_to - look_azimuth) # relative angle between wind direction and satellite heading

    phi_rel = (phi_rel + 180.0) % 360.0 - 180.0 # [-180°, 180°)

    def to_db(x):
        return 10.0 * np.log10(np.maximum(x, 1e-12))

    # CMOD5-N
    sigma0_pred = calc_sigma0_cmod5_n(v, phi_rel, theta) #Using empirical equation
    # Guard against numerical problems.
    sigma0_pred = np.asarray(sigma0_pred, dtype=np.float64)
    sigma0_pred = np.maximum(sigma0_pred, 1e-12)
    res_vv = sigma0_pred - sigma0_vv
    # res_vv = to_db(sigma0_pred) - to_db(sigma0_vv)
    
    # CMOD5-N
    # sigma0_pred2 = calc_sigma0_s1w_nr(v, theta) #Using empirical equation
    # sigma0_pred2 = np.maximum(sigma0_pred2, 1e-12)
    # res_vh = to_db(sigma0_pred2) - to_db(sigma0_vh)

    return np.concatenate([np.ravel(sqrt_weights*res_vv)]) #, np.ravel(0.1*sqrt_weights*res_vh)])