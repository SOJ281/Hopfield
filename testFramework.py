from hopfield import Hopfield, ContinuousHopfield, DAMDiscreteHopfield
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import time

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
        blocked[i] = 1
    return blocked


class HopfieldBinary:
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):

        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.empty((self.n,self.n)) #Connection matrix
        self.X = np.array(inputs)

        #w(ij) ​= ∑p​[s(i)​(p) * s(j)​(p)]
        for i in range(self.itemsLen):
            self.weights += np.outer(2*inputs[i]-1, 2*inputs[i]-1)

        for i in range(self.n):
            self.weights[i][i] = 0
        # notsure if this step is necessary
        self.weights = self.weights/self.itemsLen

    #Prediction function
    #yini ​​= xi​ + ∑j​[yj * ​wji​]
    #yi = yini{1 if > =0, else 0}
    #Iterates until hits iteration count or energy minimized
    def predict(self, input, iterations, theta = 0.0):
        #print("Predictions")

        predicted = [np.copy(input)]

        s = self.energy(predicted[0])

        for i in range(0, iterations):
            newVal = np.sign(self.weights @ predicted[i])

            st = self.energy(newVal, theta = theta)
            if s == st:
                break
            s = st
            predicted.append(newVal)
        return predicted
    
    #Prediction function
    #yini ​​= xi​ + ∑j​[yj * ​wji​]
    #yi = yini{1 if > =0, else 0}
    #Iterates until hits iteration count or energy minimized
    def predictAsyn(self, input, iterations, theta = 0.0):
        #print("Predictions")

        predicted = [np.copy(input)]
        
        for l in range(iterations):
            valList = np.arange(0, self.n)
            random.shuffle(valList)


            vals = predicted[l].copy()
            noFlip = True

            prev = self.energy(vals)

            for i in valList:
                new_vals = vals.copy()
                new_vals[i] *= -1

                current = self.energy(new_vals, theta)

                if (current - prev) < 0:
                    prev = self.energy(new_vals, theta)
                    vals[i] = new_vals[i]
                    noFlip = False
            if noFlip:
                break
            predicted.append(vals)
        return predicted

    #E = 0.5 * ∑i​∑j[​wij​ * vi * ​vj] ​+ ∑i[​θi​ * vi]
    def energy(self, state, theta = 0.0):
        return -0.5 * state @ self.weights @ state + np.sum(state*theta)


"""
for qxlp in range (15, 30, 1):
    rater = []
    errorCounter = 0
    counter = 0
    for i in range(0, 100, 1):
        patterns = np.array([random.choices([-1,1], k=100) for p in range(15)])
        hoppy = Hopfield(patterns)

        corrupted = [highBlocking(d, 0.4) for d in patterns]

        predictions = []
        for p in range(len(corrupted)):
            predictions.append(hoppy.predict(corrupted[p], 15)[-1])
        
        #print(i, ":", (patterns==predictions).sum()/(l*i))
        #print((patterns==predictions).sum()/(l*i))
        #print(l)

        counter +=1
        rater.append(np.mean( patterns != predictions ))
        if np.mean( patterns != predictions ) < 1.0:
            errorCounter+=1
            #print("Error")

    #print(errorCounter/counter)
    #print(np.mean(rater))
    print("Patterns: ",qxlp, errorCounter/counter, np.mean(rater))

exit(1)
"""


def GeneralErrorstuff():
    print("============================================")
    print("General Error stuff")
    print("============================================")


    file = open("Results/HopfieldErrorCorruption.csv",'w')
    file.write("Pattern,ErrorRate,CorruptionLevel\n")

    #from numpy import random
    for patternCount in range(1, 75):
        rater = []
        errorCounter = 0
        counter = 0
        correctRatio = []
        errorRate = []
        min_corruption = 0
        max_corruption = 50
        corruption_step = 10
        num_neurons = 100
        patternSize = 100 # Should be equal to the number of neurons?
        predict_iterations = 100
        for corruption_level in range(min_corruption,max_corruption+corruption_step,corruption_step):
            for purple in range(0, 50, 1):
                patterns = np.array([random.choices([-1,1], k=num_neurons) for p in range(patternCount)])
                #patterns = np.random.choice([0,1], size=(patternCount, 100))
                hoppy = Hopfield(patterns)

                corrupted = [randomFlipping(d, (corruption_level/100)) for d in patterns]
                #print(corrupted)

                predictions = []
                for p in range(len(corrupted)):
                    predictions.append(hoppy.predict(corrupted[p], predict_iterations)[-1])
                    # [-1] returns the final prediction (after predict_iteration iterations)

                counter +=1

                errorRate.append(np.mean((patterns != predictions).sum(1)) / num_neurons)

            #print(errorCounter/counter)
            #print(np.mean(rater))
            
            print("Patterns: ",patternCount, np.mean(errorRate),(corruption_level/100))
            file.write("%s,%s,%s\n" % (patternCount, np.mean(errorRate), (corruption_level/100)))
        
    
    file.close()


def HopfieldSyncTests():
    print("============================================")
    print("Hopfield")
    print("============================================")

    file = open("Results/HopfieldSync.csv",'w')
    file.write("Patterns,Neurons,Ratio\n")

    patternPoints = []
    neuronPoints = []
    for l in range(10, 70, 1): ## no. of Patterns
        counter = 0
        for i in range(40, 100000, 5): # no. of Neurons
            try:
                patterns = np.array([random.choices([-1,1], k=i) for p in range(l)])
                hoppy = Hopfield(patterns)

                corrupted = [highBlocking(d, 0.4) for d in patterns]

                predictions = []
                for p in range(len(corrupted)):
                    predictions.append(hoppy.predict(corrupted[p], 10)[-1])
                
                if (np.mean( patterns != predictions ) > 0.5):
                    counter +=1
                    patternPoints.append(l)
                    neuronPoints.append(i)
                    print("patterns: ", len(patterns), "Neurons: ", len(patterns[0]), "Ratio: ", l/i)
                    file.write("%s,%s,%s\n" % (len(patterns),len(patterns[0]),l/i))

                if (counter > 0):
                    break
            except:
                pass


    fig, ax = plt.subplots(2,2)
    ax[0, 0].scatter(patternPoints, neuronPoints)
    ax[0, 0].set_title("Hopfield") 

    file.close()

def HopfieldAsyncTests():
    file = open("Results/HopfieldAsync.csv",'w')
    file.write("Patterns,Neurons,Ratio\n")

    patternPoints = []
    neuronPoints = []
    for l in range(10, 70, 5): ## no. of Patterns
        counter = 0
        for i in range(40, 100000, 20): # no. of Neurons
            try:
                patterns = np.array([random.choices([-1,1], k=i) for p in range(l)])
                hoppy = Hopfield(patterns)

                corrupted = [highBlocking(d, 0.4) for d in patterns]

                predictions = []
                for p in range(len(corrupted)):
                    predictions.append(hoppy.predictAsyn(corrupted[p], 10)[-1])
                    #predictions.append(hoppy.predict(corrupted[p], 10)[-1])
                
                #print(i, ":", (patterns==predictions).sum()/(l*i))
                #print((patterns==predictions).sum()/(l*i))
                #print(l)
                if (np.mean( patterns == predictions ) == 1):
                #if (np.mean( patterns != predictions ) > 0.5):
                    counter +=1
                    patternPoints.append(l)
                    neuronPoints.append(i)
                    print("patterns: ", len(patterns), "Neurons: ", len(patterns[0]), "Ratio: ", l/i)
                    file.write("%s,%s,%s\n", (len(patterns),len(patterns[0]),l/i))

                if (counter > 0):
                    break
            except:
                pass


    fig, ax = plt.subplots(2,2)
    ax[0, 1].scatter(patternPoints, neuronPoints)
    ax[0, 1].set_title("HopfieldAsyn")

    file.close()


def DAMTests():
    print("============================================")
    print("Dense Associative Memory")
    print("============================================")

    file = open("Results/DAM.csv",'w')
    file.write("Patterns,Neurons,Ratio\n")

    dpatternPoints = []
    dneuronPoints = []
    for l in range(10, 70, 10): # no. of Patterns
        counter = 0
        for i in range(40, 100000, 20): # no. of Neurons
            try:
                patterns = np.array([random.choices([-1,1], k=i) for p in range(l)])
                hoppy = DAMDiscreteHopfield(patterns)

                corrupted = [highBlocking(d, 0.4) for d in patterns]

                predictions = []
                for p in range(len(corrupted)):
                    predictions.append(hoppy.predict(corrupted[p], 1)[-1])
                
                #print(i, ":", (patterns==predictions).sum()/(l*i))
                #print((patterns==predictions).sum()/(l*i))
                #print(l)
                if ((patterns==predictions).sum()/(l*i) == 0.5):
                    counter +=1
                    dpatternPoints.append(l)
                    dneuronPoints.append(i)
                    print("patterns: ", l, "Neurons: ", i, "Ratio: ", l/i)
                    file.write("%s,%s,%s\n" % (l,i,l/i))

                if (counter > 2):
                    break
            except:
                pass

    plt.title("DAM")
    plt.xlabel("pattern number")
    plt.ylabel("Neurons")
    figb, axb = plt.subplots()
    axb[1, 0].scatter(dpatternPoints, dneuronPoints)
    axb[1, 0].set_title("DAM") 
    axb.title("DAM")
    plt.show()

    file.close()


if __name__ == '__main__':
    GeneralErrorstuff()
    
    # HopfieldSyncTests()
    # HopfieldSyncTests()
    # DAMTests()