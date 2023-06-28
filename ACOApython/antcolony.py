import util
import random
import copy
import math

class ac:

    def __init__(self,  citylist) -> None:
        self.cost = list()
        self.phero = list()
        self.nextphero = list()
        self.cl = list(citylist)
        self.cldim = len(citylist)
        self.alpha = 1
        self.beta = 1
        self.bestwaysofar = list()
        self.lenofbestwaysofar = 0
        self.generateMatrixes()

    def generateMatrixes(self) -> None:
        self.cost = [[0] * self.cldim for _ in range(self.cldim)]
        for x in range(self.cldim):
            for y in range(x+1,self.cldim):
                way = util.calculateway(self.cl[x],self.cl[y])
                self.cost[x][y] = way
                self.cost[y][x] = way
        self.phero = [[1] * self.cldim for _ in range(self.cldim)]
        for x in range(self.cldim):
            self.phero[x][x] = 0
        self.nextphero = [[0] * self.cldim for _ in range(self.cldim)]
        self.bestwaysofar = list(range(self.cldim))
        self.lenofbestwaysofar = self.calulate_way_from_route(self.bestwaysofar)

    def doIteration(self, anzAnts=2000, p=0.5) -> None:
        for _ in range(anzAnts):
            self.oneanttourconstruction()
        self.phermoneupdate(p)

    def oneanttourconstruction(self) -> None:
        random.seed()
        acl = list(range(self.cldim)) # aktuelle city-list
        route = list()
        route.append(acl.pop(random.randrange(0,self.cldim-1))) # start city
        while len(acl) > 1:
            probabilities = self.calculate_probabilities_for_next_step(acl, route[-1]) # route[-1] ist die aktuelle city #berechent liste der einzellnen Wahrschienlichkeiten
            sum_of_probabilities = sum(probabilities) # Summe aller Wahrscheinlichkeiten
            for k in range(len(probabilities)):
                probabilities[k] = probabilities[k] / sum_of_probabilities # Normiert die Wahrscheinlichkeiten
            for x in range(len(probabilities)):
                for y in range(x+1, len(probabilities)-1):
                    probabilities[x] += probabilities[y] # Kumuliert die genormten Wahrscheinlichkeiten
            rand = random.random()
            i = 0
            while rand <= probabilities[i+1]:
                i += 1
                if i > len(probabilities)-2:
                    break
            route.append(acl.pop(i))
        route.append(acl.pop(0))
        self.update_next_cost_matrix(route)

    def update_next_cost_matrix(self, route) -> None:
        way = self.calulate_way_from_route(route)
        if way < self.lenofbestwaysofar:
            self.lenofbestwaysofar = way
            self.bestwaysofar = copy.deepcopy(route)
        nway = 1/way
        for i in range(len(route)):
            if i == 0:
                self.nextphero[route[0]][route[-1]] += nway
                self.nextphero[route[-1]][route[0]] += nway
            else:
                self.nextphero[route[i-1]][route[i]] += nway
                self.nextphero[route[i]][route[i-1]] += nway

    def calulate_way_from_route(self, route) -> None:
        way = 0.0
        for i in range(len(route)):
            if i == 0:
                way += self.cost[route[0]][route[-1]]
            else:
                way += self.cost[route[i-1]][route[i]]
        return way

    def calculate_probabilities_for_next_step(self, acl, actc) -> list:
        probabilities = list()
        for city in acl:
            probabilities.append(self.calculate_probabilitie(actc,city))
        return probabilities
    
    def calculate_probabilitie(self, c1, c2) -> float:
        return math.pow(self.phero[c1][c2],self.alpha) * math.pow(1/self.cost[c1][c2],self.beta)

    def phermoneupdate(self, p) -> None:
        for x in range(self.cldim):
            for y in range(x+1,self.cldim):
                np = (1-p)
                self.phero[x][y] = self.phero[x][y] * np
                self.phero[y][x] = self.phero[y][x] * np
        for x in range(self.cldim):
            for y in range(self.cldim):
                self.phero[x][y] += self.nextphero[x][y]
                self.nextphero[x][y] = 0

    def getcost (self) -> list:
        return self.cost
    
    def getphero (self) -> list:
        return self.phero
    
    def getbestroute (self) -> list:
        return self.bestwaysofar
    
    def getbestroutelen (self) -> float:
        return self.lenofbestwaysofar


"""
cl0 = ((0, 0), (0, 1), (1, 1), (1, 0), (2, 1))
cl1 = ((10,10),(10,20),(20,20))
cl2 = ((100,100),(100,200),(200,200),(150,150))
cl3 = ((50, 50), (50, 55), (50, 60), (50, 65), (50, 70))
cl4 = ((182,663),(232,33),(230,787),(370,676),(256,996),(600,247),(33,672),(119,225),(525,985),(716,397))
cl5 = ((50, 50), (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90), (50, 95), (50, 100))

aco = ac(cl4)
for _ in range(50):
    aco.doIteration()
    br = aco.getbestroute()
    print(br)
    brl = aco.getbestroutelen()
    print(brl)
"""
"""(3, 8, 4, 2, 0, 6, 7, 1, 5, 9)"""
"""[1, 7, 6, 0, 2, 4, 8, 3, 9, 5]"""
