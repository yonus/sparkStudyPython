from numpy import random
from sklearn.metrics import r2_score
total = {20:0,30:0, 40:0 , 50:0 , 60:0 , 70:0}
purchases = {20:0,30:0, 40:0 , 50:0 , 60:0 , 70:0}
chooseList = [20,30,40,50,60,70]
totalPurchases = 0;

for _ in range(100000):
    ageDecade = random.choice(chooseList)
    purchaseProbability = float(ageDecade)/100
    total[ageDecade] += 1;
    if(random.random() < purchaseProbability):
        totalPurchases += 1;
        purchases[ageDecade] += 1;


def calculateProbalitityGiven(x):
    return float(purchases[x])/float(total[x])

for x in chooseList:
    print(calculateProbalitityGiven(x))


