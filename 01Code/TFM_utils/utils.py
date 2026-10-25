import math


def remove_nan_wind(wind_speeds, heights, weights):
    for pos in range(len(wind_speeds) - 1, -1, -1):
        if math.isnan(wind_speeds[pos]):
            wind_speeds.pop(pos)
            heights.pop(pos)
            weights.pop(pos)

    return wind_speeds, heights, weights

def return_stability(L):
    if abs(L)>500:
        if L>500:
            L_bounds = [-1000, -500]
            stability = "nn"
        else:
            L_bounds = [500, 1000]
            stability = "np"
    elif L<0:
        L_bounds = [-500,-1]
        stability = "u"
    elif L>0:
        L_bounds = [1,500]
        stability = "s"
    return stability, L_bounds