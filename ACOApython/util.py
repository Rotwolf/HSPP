import math

def calculateway(p1,p2) -> float:
    x = p1[0]-p2[0]
    y = p1[1]-p2[1]
    return round(math.sqrt((x*x)+(y*y)))