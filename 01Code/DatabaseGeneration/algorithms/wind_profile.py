import numpy as np

def diabatic_wind_profile(z, u_star, L, stability = 1):
        """
        Calculate the diabatic vertical wind profile using MOST
        and the Businger-Dyer stability functions.

        Parameters
        ----------
        z : float or array-like
            Height above the sea surface [m].
        u_star : float
            Friction velocity [m/s], must be positive.
        L : float
            Obukhov length [m].
            L > 0: stable; L < 0: unstable.
            For neutral conditions, use np.inf.
        stability : {"u", "n", "s", 0, 1, 2}
            Atmospheric stability:
            0 or "u": unstable
            1 or "n": neutral
            2 or "s": stable

        Returns
        -------
        U : float or numpy.ndarray
            Mean horizontal wind speed [m/s].
        """
        # 0. Constants:
        alpha_c = 0.012
        kappa=0.4
        beta=6.0
        gamma=19.3
        g=9.81
        
        # 1. Validate inputs
        z_array = np.asarray(z, dtype=float)
        scalar_input = z_array.ndim == 0

        if not np.all(np.isfinite(z_array)) or np.any(z_array <= 0):
            raise ValueError("All heights z must be finite and > 0.")

        if not np.isfinite(u_star) or u_star <= 0:
            raise ValueError("u_star must be finite and > 0.")

        if np.isnan(L):
            raise ValueError("L cannot be NaN.")

        # 2. Normalize stability convention
        stability_map = {
            0: "u",
            1: "n",
            2: "s",
            "u": "u",
            "n": "n",
            "s": "s",
        }

        try:
            stability = stability_map[stability]
        except (KeyError, TypeError):
            raise ValueError(
                "stability must be 0/'u' (unstable), "
                "1/'n' (neutral), or 2/'s' (stable)."
            )

        # 3. Charnock roughness length
        z0 = alpha_c * u_star**2 / g

        # 4. Stability correction Psi_m(z/L)
        if stability == "n":
            # Neutral limit: L -> +/- infinity
            psi_m = np.zeros_like(z_array)

        elif stability == "s":
            if not np.isfinite(L) or L <= 0:
                raise ValueError(
                    "Stable conditions require a finite L > 0."
                )

            zeta = z_array / L
            psi_m = -beta * zeta

        else:  # Unstable
            if not np.isfinite(L) or L >= 0:
                raise ValueError(
                    "Unstable conditions require a finite L < 0."
                )

            zeta = z_array / L
            x = (1.0 - gamma * zeta)**0.25

            psi_m = (
                2.0 * np.log((1.0 + x) / 2.0)
                + np.log((1.0 + x**2) / 2.0)
                - 2.0 * np.arctan(x)
                + np.pi / 2.0
            )

        # 5. Diabatic wind profile
        U = (u_star / kappa) * (
            np.log(z_array / z0) - psi_m
        )

        if scalar_input:
            return float(U)

        return U