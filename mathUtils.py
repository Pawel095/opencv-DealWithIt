def get_dist(e):
    """returns distance between 2 points passed as an array"""
    import math
    return math.sqrt(pow(e[0][0] - e[1][0], 2) + pow(e[0][1] - e[1][1], 2))
