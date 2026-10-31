import numpy as np

ALPHA_C = 0.012
KAPPA=0.4
BETA=6.0
GAMMA=19.3
G=9.81
STABILITY_MAP = {
    0: "u", "u": "u", "nu": "u",
    1: "n", "n": "n",
    2: "s", "s": "s", "ns": "s"
}

def diabatic_wind_profile(z, u_star, obu_parameter, z_ref: float = 60.0, scaled: bool = False, stability = 1):
        # 0. Obukhov parameter means L or eta
        
        # 1. Validate inputs
        z_array = np.asarray(z, dtype=float)

        # 2. Normalize stability convention
        stability = STABILITY_MAP[stability]
        
        # 3. Charnock roughness length
        z0 = ALPHA_C * u_star**2 / G

        # 4. Stability parameter
        if scaled:
            stability_z = (z_array/z_ref) * obu_parameter
            stability_0 = (z0/z_ref) * obu_parameter
            
        else: 
            stability_z = z_array / obu_parameter
            stability_0 = z0 / obu_parameter

        # 5. Stability correction Psi_m(z/L)
        if stability == "n":
            # Neutral limit: L -> +/- infinity
            psi_m = np.zeros_like(z_array)
            psi_m2 = np.zeros_like(z_array)

        elif stability == "s": # Stable
            psi_m = -BETA * stability_z
            psi_m2 = -BETA * stability_0

        else:  # Unstable
            x = (1.0 - GAMMA * stability_z)**0.25 
            x2 = (1.0 - GAMMA * stability_0)**0.25

            psi_m = (2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x**2) / 2.0) - 2.0 * np.arctan(x) + np.pi / 2.0)
            psi_m2 = (2.0 * np.log((1.0 + x2) / 2.0) + np.log((1.0 + x2**2) / 2.0) - 2.0 * np.arctan(x2) + np.pi / 2.0)

        # 6. Diabatic wind profile
        U = (u_star / KAPPA) * (np.log(z_array / z0) - psi_m + psi_m2)
        return U

def delta_diabatic_wind_profile(z2, z1, u_star, obu_parameter, z_ref: float = 60.0, scaled: bool = False, stability = 1):
        # 0. Obukhov parameter means L or eta
        
        # 1. Validate inputs
        z2_array = np.asarray(z2, dtype=float)
        z1_array = np.asarray(z1, dtype=float)

        # 2. Normalize stability convention
        stability = STABILITY_MAP[stability]

        # 3. Stability parameter
        if scaled:
            stability_z2 = (z2_array/z_ref) * obu_parameter
            stability_z1 = (z1_array/z_ref) * obu_parameter
            
        else: 
            stability_z2 = z2_array / obu_parameter
            stability_z1 = z1_array / obu_parameter
                    
        # 4. Stability correction Psi_m(z/L)
        if stability == "n":
            # Neutral limit: L -> +/- infinity
            psi_m2 = np.zeros_like(z2_array)
            psi_m1 = np.zeros_like(z1_array)

        elif stability == "s":
            psi_m2 = -BETA * stability_z2
            psi_m1 = -BETA * stability_z1

        else:  # Unstable
            x2 = (1.0 - GAMMA * stability_z2)**0.25 
            x1 = (1.0 - GAMMA * stability_z1)**0.25

            psi_m2 = (2.0 * np.log((1.0 + x2) / 2.0) + np.log((1.0 + x2**2) / 2.0) - 2.0 * np.arctan(x2) + np.pi / 2.0)
            psi_m1 = (2.0 * np.log((1.0 + x1) / 2.0) + np.log((1.0 + x1**2) / 2.0) - 2.0 * np.arctan(x1) + np.pi / 2.0)

        # 5. Diabatic wind profile
        diff_U = (u_star / KAPPA) * (np.log(z2_array / z1_array) - psi_m2 + psi_m1)
        return diff_U