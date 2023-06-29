import citys
import bruteforce
import pygame
import antcolony
from time import sleep
import os

def read_tsp_file(filename):
    filename = "C:/Users/jonie/Desktop/hausarbeit/testtsps/" + filename +".tsp"
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

dj38 = read_tsp_file("dj38")
soldj38 = [12, 14, 19, 22, 25, 24, 21, 23, 27, 26, 30, 35, 33, 32, 37, 36, 34, 31, 29, 28, 20, 13, 9, 0, 1, 3, 2, 4, 5, 6, 7, 8, 11, 10, 18, 17, 16, 15] #[21, 23, 27, 26, 30, 35, 33, 32, 37, 36, 34, 31, 29, 28, 20, 13, 9, 0, 1, 3, 2, 4, 5, 6, 7, 8, 11, 10, 18, 17, 16, 15, 12, 14, 19, 22, 25, 24]  

lu980 = read_tsp_file("lu980")
sollu980 = list(range(len(lu980)))

qa194 = read_tsp_file("qa194")
solqa194a = [161, 157, 158, 164, 167, 166, 169, 179, 177, 176, 174, 172, 173, 178, 171, 168, 163, 162, 160, 155, 144, 148, 145, 137, 141, 136, 139, 131, 126, 124, 125, 133, 129, 110, 103, 100, 98, 93, 88, 89, 97, 85, 84, 64, 19, 62, 35, 61, 58, 81, 79, 70, 75, 71, 73, 77, 74, 90, 101, 108, 112, 118, 121, 117, 130, 128, 134, 132, 142, 147, 135, 154, 150, 146, 151, 152, 149, 143, 140, 156, 153, 138, 175, 181, 182, 186, 185, 193, 189, 191, 187, 183, 180, 188, 190, 192, 184, 170, 165, 159, 127, 123, 122, 119, 120, 116, 115, 114, 111, 109, 107, 106, 104, 105, 96, 94, 95, 92, 87, 82, 67, 65, 66, 72, 26, 33, 38, 46, 50, 57, 42, 39, 37, 40, 48, 41, 34, 30, 31, 29, 49, 54, 53, 51, 52, 47, 45, 43, 18, 14, 11, 9, 8, 4, 2, 6, 10, 12, 15, 7, 5, 3, 1, 0, 22, 24, 13, 16, 25, 23, 20, 17, 32, 59, 68, 76, 69, 63, 56, 44, 36, 28, 21, 27, 60, 55, 99, 83, 78, 80, 91, 102, 86, 113]
solqa194b = [69, 63, 56, 44, 36, 38, 33, 42, 39, 37, 30, 31, 29, 34, 41, 48, 54, 53, 51, 52, 47, 45, 40, 43, 55, 46, 57, 60, 72, 67, 65, 66, 50, 26, 21, 27, 32, 59, 68, 73, 71, 74, 77, 86, 79, 70, 75, 81, 61, 58, 35, 62, 19, 64, 84, 85, 97, 89, 88, 93, 98, 100, 103, 110, 113, 118, 112, 108, 101, 102, 90, 87, 82, 78, 80, 76, 83, 99, 109, 111, 114, 115, 116, 120, 119, 122, 123, 127, 132, 128, 134, 135, 130, 150, 154, 147, 142, 159, 165, 161, 157, 158, 164, 167, 177, 179, 184, 192, 187, 190, 191, 189, 186, 185, 182, 173, 172, 178, 171, 168, 175, 181, 193, 188, 183, 174, 180, 176, 169, 170, 166, 151, 146, 140, 143, 149, 152, 156, 153, 138, 137, 141, 145, 148, 144, 139, 136, 133, 131, 126, 124, 125, 129, 155, 160, 162, 163, 105, 106, 107, 104, 95, 94, 91, 96, 92, 117, 121, 49, 18, 14, 11, 9, 8, 4, 2, 6, 10, 16, 13, 12, 22, 24, 15, 7, 5, 3, 1, 0, 23, 25, 20, 17, 28]
solqa194c = [139, 136, 141, 145, 155, 144, 148, 160, 162, 163, 168, 171, 178, 173, 172, 185, 186, 189, 193, 175, 181, 174, 182, 183, 188, 190, 191, 187, 192, 177, 179, 184, 170, 165, 161, 157, 158, 146, 151, 152, 149, 143, 140, 150, 135, 132, 134, 128, 130, 121, 118, 124, 125, 137, 138, 153, 156, 166, 167, 164, 180, 176, 169, 159, 154, 147, 142, 127, 123, 122, 119, 120, 116, 115, 114, 111, 109, 107, 106, 104, 105, 95, 94, 91, 87, 82, 78, 80, 83, 76, 69, 59, 56, 63, 67, 65, 66, 55, 34, 37, 40, 45, 53, 51, 52, 47, 57, 60, 50, 46, 38, 36, 28, 21, 27, 32, 17, 20, 23, 25, 16, 13, 10, 12, 22, 24, 70, 81, 88, 89, 93, 98, 100, 103, 110, 97, 85, 84, 64, 19, 62, 35, 58, 61, 15, 7, 5, 0, 1, 3, 6, 2, 4, 8, 9, 11, 14, 18, 49, 41, 43, 48, 54, 39, 33, 30, 31, 29, 26, 72, 96, 92, 90, 102, 101, 86, 79, 75, 71, 74, 77, 73, 68, 44, 42, 99, 117, 112, 108, 113, 129, 126, 131, 133]
solqa194d = [49, 41, 51, 52, 47, 45, 54, 48, 43, 37, 42, 55, 57, 46, 50, 38, 33, 39, 30, 26, 36, 60, 65, 67, 63, 69, 76, 78, 80, 82, 91, 87, 92, 95, 94, 96, 105, 104, 106, 107, 109, 111, 114, 115, 116, 120, 119, 127, 123, 122, 159, 165, 161, 157, 158, 164, 167, 169, 166, 150, 154, 147, 142, 132, 128, 134, 135, 130, 117, 121, 118, 112, 108, 113, 125, 124, 155, 144, 148, 145, 141, 136, 139, 137, 138, 143, 140, 151, 146, 149, 152, 156, 153, 163, 168, 175, 181, 171, 178, 173, 172, 174, 176, 180, 177, 179, 184, 170, 192, 187, 183, 188, 190, 191, 189, 186, 185, 182, 193, 160, 162, 131, 133, 126, 129, 110, 103, 100, 98, 93, 89, 88, 97, 85, 84, 64, 19, 62, 35, 58, 61, 81, 70, 24, 22, 12, 15, 7, 5, 0, 1, 3, 6, 10, 13, 16, 25, 23, 20, 17, 27, 21, 28, 44, 56, 59, 68, 73, 77, 74, 71, 75, 86, 79, 102, 101, 90, 99, 83, 72, 66, 31, 29, 34, 53, 40, 18, 14, 11, 9, 8, 4, 2, 32]

a280 = read_tsp_file("a280")
sola280 = list(range(len(a280)))

d198 = read_tsp_file("d198")
sold198 = list(range(len(d198)))

lin318 = read_tsp_file("lin318")
sollin318 = list(range(len(lin318)))

pcb442 = read_tsp_file("pcb442")
solpcb442 = list(range(len(pcb442)))

pr1002 = read_tsp_file("pr1002")
solpr1002 = list(range(len(pr1002)))

rat783 = read_tsp_file("rat783")
solrat783 = list(range(len(rat783)))


cl4 = ((182,663),(232,33),(230,787),(370,676),(256,996),(600,247),(33,672),(119,225),(525,985),(716,397))
solcl4 = [1, 7, 6, 0, 2, 4, 8, 3, 9, 5]
"""
region = citys.citys()
wasgenerationsucessfull = region.clearandautogenerate(100, dis=5)
if not wasgenerationsucessfull:
    print("generation failed")
    exit()
clist = region.getlist()
"""
clist = qa194
#bsol = bruteforce.brutforce(clist)
#aco = antcolony.ac(clist)
#aco.doIteration()
#sol = aco.getbestroute()
sol = solqa194d

min_x = min(point[0] for point in clist)
max_x = max(point[0] for point in clist)
min_y = min(point[1] for point in clist)
max_y = max(point[1] for point in clist)

# Scale the points to the range [0, 1000]
scaled_list = [[((point[0] - min_x) / (max_x - min_x)) * 1000, ((point[1] - min_y) / (max_y - min_y)) * 1000] for point in clist]

pygame.init()
pygame.display.set_caption("ACOA")

screen = pygame.display.set_mode((1100,1100))

background_color = (0, 0, 0)

i = 0
running = True
while running:

    screen.fill(background_color)

    for city in scaled_list:
        pygame.draw.circle(screen,(255, 0, 0), (city[0]+50,city[1]+50), 4)

    for k in range(len(sol)):
        if k == 0:
            pygame.draw.line(screen,(255,255,255), (scaled_list[sol[0]][0]+50,scaled_list[sol[0]][1]+50), (scaled_list[sol[-1]][0]+50,scaled_list[sol[-1]][1]+50),2)
        else:
            pygame.draw.line(screen,(255,255,255), (scaled_list[sol[k]][0]+50,scaled_list[sol[k]][1]+50), (scaled_list[sol[k-1]][0]+50,scaled_list[sol[k-1]][1]+50),2)

    #for k in range(len(bsol)):
    #    if k == 0:
    #        pygame.draw.line(screen,(0,0,255), (scaled_list[bsol[0]][0]+50,scaled_list[bsol[0]][1]+50), (scaled_list[bsol[-1]][0]+50,scaled_list[bsol[-1]][1]+50),1)
    #    else:
    #        pygame.draw.line(screen,(0,0,255), (scaled_list[bsol[k]][0]+50,scaled_list[bsol[k]][1]+50), (scaled_list[bsol[k-1]][0]+50,scaled_list[bsol[k-1]][1]+50),1)

    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect((i % 1000)+50, (i % 1000)+50, 40, 30))
    i += 1

    #aco.doIteration(500)
    #sol = aco.getbestroute()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    sleep(0.1)
