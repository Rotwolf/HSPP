def check_duplicates(numbers):
    return len(numbers) != len(set(numbers))

def find_missing_numbers(numbers):
    max_num = max(numbers)
    missing_numbers = set(range(max_num + 1)) - set(numbers)
    return sorted(missing_numbers)
"""
numbers = [191, 190, 188, 187, 183, 182, 185, 186, 189, 193, 181, 175, 168, 163, 162, 160, 155, 131, 133, 129, 126, 124, 125, 113, 112, 108, 101, 102, 90, 82, 80, 78, 76, 69, 63, 56, 44, 36, 26, 21, 28, 27, 32, 17, 20, 23, 25, 16, 13, 10, 12, 22, 24, 15, 7, 5, 0, 1, 3, 6, 2, 4, 8, 9, 11, 14, 18, 29, 31, 34, 37, 40, 45, 47, 52, 51, 53, 54, 48, 41, 43, 49, 55, 57, 46, 38, 33, 39, 42, 30, 50, 60, 66, 72, 65, 67, 59, 68, 73, 71, 74, 77, 75, 70, 79, 86, 81, 61, 58, 35, 62, 19, 64, 84, 85, 97, 89, 93, 88, 98, 100, 103, 110, 118, 121, 117, 107, 106, 104, 105, 96, 94, 95, 92, 87, 91, 83, 99, 109, 111, 114, 115, 116, 120, 119, 122, 123, 127, 132, 134, 128, 130, 135, 142, 147, 154, 150, 146, 151, 140, 143, 149, 153, 156, 152, 138, 137, 141, 136, 139, 145, 148, 144, 178, 171, 172, 173, 174, 176, 180, 177, 179, 184, 170, 169, 166, 167, 164, 158, 157, 161, 165, 159, 192]

has_duplicates = check_duplicates(numbers)
missing_numbers = find_missing_numbers(numbers)

print("Has duplicates:", has_duplicates)
print("Missing numbers:", missing_numbers)
"""
def read_tsp_file(filename):
    coordinates = {}
    with open("C:/Users/jonie/Desktop/hausarbeit/testtsps/" + filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if line == "EOF":
                break
            if line.startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD_SECTION"
                continue
            if section == "NODE_COORD_SECTION":
                if line:
                    index, x, y = map(float, line.split())
                    if (x, y) in coordinates.values():
                        print(f"Duplicate coordinates found for index {index}: ({x}, {y})")
                    else:
                        coordinates[index] = (x, y)
    return coordinates


filename = "lu980.tsp"  # Replace with your .tsp file name
coordinate_data = read_tsp_file(filename)
print("Duplicate Coordinates:")
for index, (x, y) in coordinate_data.items():
    print(f"Index: {index}, Coordinates: ({x}, {y})")


