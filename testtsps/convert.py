import os

def read_tsp_file(filename):
    filename = "C:/Users/jonie/Desktop/hausarbeit/testtsps/" + filename
    with open(filename, 'r') as file:
        lines = file.readlines()

    coordinates = []
    i = 0
    for line in lines:
        if line.strip() == "EOF":
            break
        if i == 0:
            if line.strip() == "NODE_COORD_SECTION":
                i = 1
        else:
            s, x, y = line.strip().split()
            coordinates.append([float(x),float(y)])

    return coordinates

filename = 'lu980.tsp'  # Replace
coordinates_python = read_tsp_file(filename)
print(coordinates_python)