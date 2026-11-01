from .wind_profile import diabatic_wind_profile, delta_diabatic_wind_profile
import numpy as np

def least_squares_residuals(params,
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