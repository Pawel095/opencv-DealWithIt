def get_dist(e):
    """returns distance between 2 points passed as an array"""
    import math
    return math.sqrt(pow(e[0][0] - e[1][0], 2) + pow(e[0][1] - e[1][1], 2))


def approach(val,desired,step):
    """slowly approaches a value. every execution: val will be closer to desired
    :argument val The value that will be modified approach
    :argument desired The value to approach
    :argument step (0,1] the closet this is to 1 the faster val will almost equal desired."""
    delta=(desired-val)*step
    val+=delta
    return val


def closeTo(pos1,pos2,delta):
    """check if the positions are close together"""
    if get_dist((pos1,pos2))<delta:
        return True
    else:
        return False;
