import numpy as np

def calc_sigma0_cmod5_n(v, phi, theta):
        """Calculates CMOD5n normalized radar backscatter (linear).

        Parameters:
            v: Wind speed [m/s] (>= 0)
            phi: Relative wind direction [deg] (angle between azimuth and wind
            direction)
            theta: Incidence angle [deg]

        Returns:
            CMOD5_N: Normalized backscatter sigma0 (linear scale)
        """

        # 1-based indexing added to match Fortran C(1) .. C(28)
        C = np.array(
            [
                0.0,  # Index 0 unused to maintain 1-based Fortran indexing
                -0.6878,
                -0.7957,
                0.3380,
                -0.1728,
                0.0000,
                0.0040,
                0.1103,
                0.0159,
                6.7329,
                2.7713,
                -2.2885,
                0.4971,
                -0.7250,
                0.0450,
                0.0066,
                0.3222,
                0.0120,
                22.7000,
                2.0813,
                3.0000,
                8.3659,
                -3.3428,
                1.3236,
                6.2437,
                2.3893,
                0.3249,
                4.1590,
                1.6930,
            ]
        )

        DTOR = 57.29577951
        THETM = 40.0
        THETHR = 25.0
        ZPOW = 1.6

        Y0 = C[19]
        PN = C[20]
        A = C[19] - (C[19] - 1.0) / C[20]
        B = 1.0 / (C[20] * (C[19] - 1.0) ** (3 - 1))

        # Angles
        FI = np.radians(phi)  # equivalent to phi / DTOR
        CSFI = np.cos(FI)
        CS2FI = 2.0 * CSFI * CSFI - 1.0

        X = (theta - THETM) / THETHR
        XX = X * X

        # B0: Function of wind speed and incidence angle
        A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX
        A1 = C[5] + C[6] * X
        A2 = C[7] + C[8] * X

        GAM = C[9] + C[10] * X + C[11] * XX
        S0 = C[12] + C[13] * X

        S = A2 * v
        A3 = 1.0 / (1.0 + np.exp(-np.maximum(S, S0)))

        # Piecewise condition for S < S0
        if np.ndim(S) > 0:
            mask = S < S0
            A3[mask] = A3[mask] * (S[mask] / S0[mask]) ** (
                S0[mask] * (1.0 - A3[mask])
            )
        else:
            if S < S0:
                A3 = A3 * (S / S0) ** (S0 * (1.0 - A3))

        B0 = (A3**GAM) * (10.0 ** (A0 + A1 * v))

        # B1: Function of wind speed and incidence angle
        B1 = C[15] * v * (0.5 + X - np.tanh(4.0 * (X + C[16] + C[17] * v)))
        B1 = C[14] * (1.0 + X) - B1
        B1 = B1 / (np.exp(0.34 * (v - C[18])) + 1.0)

        # B2: Function of wind speed and incidence angle
        V0 = C[21] + C[22] * X + C[23] * XX
        D1 = C[24] + C[25] * X + C[26] * XX
        D2 = C[27] + C[28] * X

        V2 = (v / V0) + 1.0

        # Piecewise condition for V2 < Y0
        if np.ndim(V2) > 0:
            mask2 = V2 < Y0
            V2[mask2] = A + B * (V2[mask2] - 1.0) ** PN
        else:
            if V2 < Y0:
                V2 = A + B * (V2 - 1.0) ** PN

        B2 = (-D1 + D2 * V2) * np.exp(-V2)

        # CMOD5_N: Combine the three Fourier terms
        CMOD5_N = B0 * (1.0 + B1 * CSFI + B2 * CS2FI) ** ZPOW

        return CMOD5_N