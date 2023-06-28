import copy
import util

def evaluate(citylist, seqlist, best) -> None:
    if seqlist[0] < seqlist[len(seqlist)-2]:
        way = 0.0
        for number in range(len(seqlist)):
            if number == 0:
                way += util.calculateway(citylist[seqlist[0]], citylist[seqlist[-1]])
            else:
                way += util.calculateway(citylist[seqlist[number]], citylist[seqlist[number-1]])
        if best[1] == -1:
            best[0] = copy.deepcopy(seqlist)
            best[1] = way
        elif way < best[1]:
            best[0] = copy.deepcopy(seqlist)
            best[1] = way
    return    

def recbruteforce(citylist, seqlist, limiter, best) -> None:
    if (limiter == 1):
        evaluate(citylist, seqlist, best)
    else:
        for i in range(limiter):
            recbruteforce(citylist, seqlist, limiter-1, best)
            if limiter % 2 == 0 :
                swapindex = i
            else: 
                swapindex = 0
            seqlist[swapindex], seqlist[limiter-1] = seqlist[limiter-1], seqlist[swapindex]
    return

def brutforce(citylist) -> tuple:
    if len(citylist) < 2 :
        return ()
    seq = list(range(len(citylist)))
    best = [-1, -1]
    recbruteforce(citylist, seq, len(seq)-1, best)
    return tuple(best[0])

"""
cl0 = ((0, 0), (0, 1), (1, 1), (1, 0))
cl1 = ((10,10),(10,20),(20,20))
cl2 = ((100,100),(100,200),(200,200),(150,150))
cl3 = ((50, 50), (50, 55), (50, 60), (50, 65), (50, 70))
cl4 = ((182,663),(232,33),(230,787),(370,676),(256,996),(600,247),(33,672),(119,225),(525,985),(716,397))
cl5 = ((50, 50), (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90), (50, 95), (50, 100))

ca = cl4
soltuple = brutforce(ca)
print(soltuple)
bes=[-1,-1]
evaluate(ca,soltuple,bes)
print(bes)
"""