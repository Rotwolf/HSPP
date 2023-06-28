import random

class citys:

    def __init__(self) -> None:
        self.citylist = ()

    def clearandautogenerate(self, numberofcitys=10, x=1000, y=1000, dis=70) -> bool:
        random.seed()
        i = 0
        while len(self.citylist) < numberofcitys:
            if i > 15:
                return False
            point=(random.randrange(x),random.randrange(y))
            iskosvaild = True
            for city in self.citylist:
                if (abs(point[1]-city[1])+abs(point[0]-city[0])) < dis:
                    iskosvaild = False
            if iskosvaild:
                self.citylist = self.citylist + (point,)
                i = 0
            else:
                i += 1
        return True
    
    def getlist(self) -> tuple[tuple[int,int]]:
        return self.citylist
    
    def printlist(self) -> None:
        for i, city in enumerate(self.citylist):
            print(f"city {i} : {city[0]}/{city[1]}")
        return

        

