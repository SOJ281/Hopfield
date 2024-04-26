from hopfield import Hopfield, ContinuousHopfield, DAMDiscreteHopfield
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np

#Randomly inverts data
def randomFlipping(input, flipCount):
    flippy = np.copy(input)
    inv = np.random.binomial(n=1, p=flipCount, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            flippy[i] = -1 * v
    return flippy

#Removes random chunk of data
def highBlocking(input, blockLevel):
    blocked = np.copy(input)

    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = -1
    return blocked



print("============================================")
print("Hopfield")
print("============================================")

patternPoints = []
neuronPoints = []
for l in range(10, 100, 5): ## no. of Patterns
    counter = 0
    for i in range(40, 100000, 20): # no. of Neurons
        patterns = np.array([random.choices([-1,1], k=i) for p in range(l)])
        hoppy = Hopfield(patterns)

        corrupted = [highBlocking(d, 0.4) for d in patterns]

        predictions = []
        for p in range(len(corrupted)):
            predictions.append(hoppy.predict(corrupted[p], 3)[-1])
        
        #print(i, ":", (patterns==predictions).sum()/(l*i))
        #print((patterns==predictions).sum()/(l*i))
        #print(l)
        if ((patterns==predictions).sum()/(l*i) == 1.0):
            counter +=1
            patternPoints.append(l)
            neuronPoints.append(i)
            print("patterns: ", l, "Neurons: ", i, "Ratio: ", l/i)

        if (counter > 5):
            break

#plt.title("Hopfield")
#plt.xlabel("pattern number")
#plt.ylabel("Neurons")
fig, ax = plt.subplots(2,2)
ax[0, 0].scatter(patternPoints, neuronPoints)
ax[0, 0].set_title("Hopfield") 

print("============================================")
print("Dense Associative Memory")
print("============================================")

dpatternPoints = []
dneuronPoints = []
for l in range(10, 100, 5): # no. of Patterns
    counter = 0
    for i in range(40, 100000, 20): # no. of Neurons
        patterns = np.array([random.choices([-1,1], k=i) for p in range(l)])
        hoppy = DAMDiscreteHopfield(patterns)

        corrupted = [highBlocking(d, 0.4) for d in patterns]

        predictions = []
        for p in range(len(corrupted)):
            predictions.append(hoppy.predict(corrupted[p], 1)[-1])
        
        #print(i, ":", (patterns==predictions).sum()/(l*i))
        #print((patterns==predictions).sum()/(l*i))
        #print(l)
        if ((patterns==predictions).sum()/(l*i) == 1.0):
            counter +=1
            dpatternPoints.append(l)
            dneuronPoints.append(i)
            print("patterns: ", l, "Neurons: ", i, "Ratio: ", l/i)

        if (counter > 5):
            break

#plt.title("DAM")
#plt.xlabel("pattern number")
#plt.ylabel("Neurons")
#figb, axb = plt.subplots()
ax[1, 0].scatter(dpatternPoints, dneuronPoints)
ax[1, 0].set_title("DAM") 
#axb.title("DAM")
plt.show()




"""
#for i in range(40, 800, 40):
for i in range(2, 40, 4):
    patterns = np.array([random.choices([-1,1], k=128) for l in range(i)])
    hoppy = DAMDiscreteHopfield(patterns)

    corrupted = [highBlocking(d, 0.4) for d in patterns]

    predictions = []
    for l in range(len(corrupted)):
        predictions.append(hoppy.predict(corrupted[l], 3)[-1])
    
    print(i, ":", (patterns==predictions).sum()/(1000*i))
"""