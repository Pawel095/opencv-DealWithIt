def get_dist(e):
    """returns distance between 2 points passed as an array"""
    import math
    return math.sqrt(pow(e[0][0] - e[1][0], 2) + pow(e[0][1] - e[1][1], 2))

def approach(val,desired,step):
    """slowly approaches a value. every execution: val+=abs(desired - val)*step
    :argument val The value that will be modified approach
    :argument desired The value to approach
    :argument step (0,1] the closet this is to 1 the faster val will almost equal desired."""
    val+=abs(desired-val)*step
    return val;