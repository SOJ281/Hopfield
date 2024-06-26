from hopfield import *
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import math

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



def GeneralErrorstuff(filename, HopfieldType, nums_neurons=[100], thetas=[0.0], corruption=[0,50,10], max_patterns=[1,50,1], betas=[8], rectified=True, powers=[2], pairwise_connections=False,max_error=1):
    print("============================================")
    print("General Error stuff")
    print("============================================")

    t = time.localtime(time.time())
    formatted_t = (''.join((str(t.tm_mday),str(t.tm_hour), str(t.tm_min))))

    file = open("Results/HopfieldError%s%s.csv" % (filename,formatted_t),'w')
    file.write("Pattern,ErrorRate,CorruptionLevel,num_neurons,param\n")

    min_corruption = corruption[0]
    max_corruption = corruption[1]
    corruption_step = corruption[2]
    patternSize = 100 # Should be equal to the number of neurons?
    predict_iterations = 100

    params = [thetas,betas,powers]
    if HopfieldType == "DAMDiscreteHopfield":
        i=2
    elif "Continuous" in HopfieldType:
        i=1
    else:
        i=0

    print("Patterns: ","errorRate","corruption_level","param","time")
    #from numpy import random
    for num_neurons in nums_neurons:
        for param in params[i]:
            #for max_pattern in max_patterns:#range(max_patterns[0], max_patterns[1], max_patterns[2]):
            for patternCount in range(max_patterns[0], max_patterns[1], max_patterns[2]):
                rater = []
                errorCounter = 0
                counter = 0
                correctRatio = []
                errorRate = []
                errorRate01 = []
                errorRate001 = []
                errorRate0001 = []
                errorRate00001 = []
                start_time = time.time()
                for corruption_level in range(min_corruption,max_corruption+corruption_step,corruption_step):
                    for purple in range(0, 50, 1):
                        if HopfieldType == "Hopfield":
                            patterns = np.array([random.choices([-1,1], k=num_neurons) for p in range(patternCount)])
                            hoppy = Hopfield(patterns)
                        elif HopfieldType == "DAMDiscreteHopfield":
                            patterns = np.array([random.choices([-1,1], k=num_neurons) for p in range(patternCount)])
                            hoppy = DAMDiscreteHopfield(patterns, rectified, power=param)
                        elif HopfieldType == "DAMEXP":
                            patterns = np.array([random.choices([-1,1], k=num_neurons) for p in range(patternCount)])
                            hoppy = DAMEXP(patterns)
                        elif HopfieldType == "ContinuousBinaryHopfield":
                            patterns = np.array([random.choices([0,1], k=num_neurons) for p in range(patternCount)])
                            hoppy = ContinuousHopfield(patterns)
                        elif HopfieldType == "ContinuousHopfield":
                            random.seed(1)
                            patterns = np.array([[random.random() for k in range(num_neurons)] for p in range(patternCount)])
                            hoppy = ContinuousHopfield(patterns)
                        elif HopfieldType == "SimplicialHopfield":
                            patterns = np.array([random.choices([-1,1], k=num_neurons) for p in range(patternCount)])
                            hoppy = SimplicialHopfield(patterns, pairwise_connections=pairwise_connections)
                        #patterns = np.random.choice([0,1], size=(patternCount, 100))

                        corrupted = [randomFlipping(d, (corruption_level/100)) for d in patterns]
                        #print(corrupted)

                        predictions = []
                        for p in range(len(corrupted)):
                            if HopfieldType == "Hopfield":
                                predictions.append(hoppy.predict(corrupted[p], predict_iterations, theta=param)[-1])
                                # [-1] returns the final prediction (after predict_iteration iterations)
                            elif HopfieldType == "DAMDiscreteHopfield":
                                predictions.append(hoppy.predict(corrupted[p], predict_iterations)[-1])
                            elif HopfieldType == "DAMEXP":
                                predictions.append(hoppy.predict(corrupted[p], predict_iterations)[-1])
                            elif HopfieldType == "ContinuousHopfield" or HopfieldType == "ContinuousBinaryHopfield":
                                predictions.append(hoppy.predict(corrupted[p], predict_iterations, beta=param)[-1])
                            elif HopfieldType == "SimplicialHopfield":
                                predictions.append(hoppy.predict(corrupted[p], predict_iterations, theta=param)[-1])

                        counter +=1

                        if HopfieldType == "ContinuousHopfield":
                            errorRate.append(np.mean((patterns != predictions).sum(1)) / num_neurons)

                            errorRate01.append(np.mean((np.isclose(patterns,predictions,atol=10**-2)).sum(1)) / num_neurons)
                            errorRate001.append(np.mean((np.isclose(patterns,predictions,atol=10**-3)).sum(1)) / num_neurons)
                            errorRate0001.append(np.mean((np.isclose(patterns,predictions,atol=10**-4)).sum(1)) / num_neurons)
                            errorRate00001.append(np.mean((np.isclose(patterns,predictions,atol=10**-5)).sum(1)) / num_neurons)
                        else:
                            errorRate.append(np.mean((patterns != predictions).sum(1)) / num_neurons)

                    #print(errorCounter/counter)
                    #print(np.mean(rater))
                    
                    print("Patterns: ",patternCount, np.mean(errorRate),(corruption_level/100),param,np.mean(errorRate01),np.mean(errorRate001),np.mean(errorRate0001),np.mean(errorRate00001))
                    file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (patternCount, np.mean(errorRate), (corruption_level/100),num_neurons,param,time.time()-start_time,np.mean(errorRate01),np.mean(errorRate001),np.mean(errorRate0001),np.mean(errorRate00001)))

                    if np.mean(errorRate) > max_error:
                        break
    
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
    #GeneralErrorstuff(filename="NoCorruptionThetas",HopfieldType="Hopfield",nums_neurons=[100],thetas=[0.1,0.2,0.4,0.8],corruption=[0,10,10],max_patterns=50)
    #GeneralErrorstuff(filename="DAM",HopfieldType="DAMDiscreteHopfield",nums_neurons=[100],thetas=[0.0],corruption=[0,50,10],max_patterns=75)
    #GeneralErrorstuff(filename="Continuous",HopfieldType="ContinuousHopfield",nums_neurons=[100],thetas=[0.0],corruption=[0,50,10],max_patterns=75)
    #GeneralErrorstuff(filename="NoCorruptionJusttheta0",HopfieldType="Hopfield",nums_neurons=[100],thetas=[0.0],corruption=[0,10,10],max_patterns=50)
    
    # Current experiments
    #GeneralErrorstuff(filename="ContinuousBinaryNoMean",HopfieldType="ContinuousHopfield",nums_neurons=[100],thetas=[0.0],betas=[8],corruption=[0,50,10],max_patterns=75)
    #GeneralErrorstuff(filename="ContinuousDifferentNinjas",HopfieldType="ContinuousHopfield",nums_neurons=[100],thetas=[0.0],betas=[8182*2*2],corruption=[0, 50, 10],max_patterns=75)



    """
    To Run
    """
    # Regular Hopfield
        # Didn't think there was much else to run here

    # Simplicial
    #GeneralErrorstuff(filename="SimplicialNopairwise", HopfieldType="SimplicialHopfield",nums_neurons=[10], pairwise_connections=False,corruption=[0,10,10],max_patterns=[1, 10, 1])
    #GeneralErrorstuff(filename="SimplicialPairwise", HopfieldType="SimplicialHopfield",nums_neurons=[10], pairwise_connections=True,corruption=[0,10,10],max_patterns=[1, 10, 1])

    # DAM
    #   Rectified polynomial energy function
    #GeneralErrorstuff(filename="DAMDifferentPowerRectified",HopfieldType="DAMDiscreteHopfield",nums_neurons=[100],powers=[1,4,8],corruption=[0,50,10],max_patterns=[5, 75, 5])
    #   Polynomial energy function
    #GeneralErrorstuff(filename="DAMDifferentPowerPolynomial",HopfieldType="DAMDiscreteHopfield",nums_neurons=[100],powers=[1,2,4,8],corruption=[0,50,10],max_patterns=[5, 75, 5], rectified=False)

    # Continuous
    #   Continuous patterns
    #GeneralErrorstuff(filename="Continuous",HopfieldType="ContinuousHopfield",nums_neurons=[100],thetas=[0.0],betas=[1,2,4,8,16,32,64],corruption=[0, 50, 10],max_patterns=75)
    #   Binary patterns
    #GeneralErrorstuff(filename="ContinuousBinary",HopfieldType="ContinuousBinaryHopfield",nums_neurons=[100],betas=[64,128,256],corruption=[0, 50, 10],max_patterns=75)
        # Rectified polynomial energy function
        # polynomial energy function

    # Continuous
        # Continuous patterns
        # Binary patterns
    
    #GeneralErrorstuff(filename="ContinuousBinary",HopfieldType="ContinuousBinaryHopfield",nums_neurons=[100],betas=[64,128,256],corruption=[0, 50, 10],max_patterns=75)
    #GeneralErrorstuff(filename="DAMEXPpow1",HopfieldType="DAMEXP",nums_neurons=[100],powers=[1],corruption=[0,50,10],max_patterns=[5, 75, 5], rectified=False)
    #GeneralErrorstuff(filename="Continuous",HopfieldType="ContinuousHopfield",nums_neurons=[100],thetas=[0.0],betas=[1,2,4,8,16,32,64],corruption=[0, 50, 10],max_patterns=75)
    #GeneralErrorstuff(filename="DAMDifferentPowerRectified",HopfieldType="DAMDiscreteHopfield",nums_neurons=[100],powers=[1,4,8],corruption=[0,50,10],max_patterns=75)

    """
    To run
    """
    #GeneralErrorstuff(filename="SimplicialHopfield1st",HopfieldType="SimplicialHopfield",nums_neurons=[5,10,15,20],corruption=[0, 10, 10],max_patterns=[5,10,15,20])
    GeneralErrorstuff(filename="DAMExponential1st",HopfieldType="DAMEXP",nums_neurons=[15],corruption=[0, 10, 10],max_patterns=[5, 160, 5])
    #GeneralErrorstuff(filename="DAMExponential2nd",HopfieldType="DAMEXP",nums_neurons=[14],corruption=[0, 10, 10],max_patterns=[90,150, 5])
    # GeneralErrorstuff(filename="SimplicialHopfield2nd",HopfieldType="SimplicialHopfield",nums_neurons=[25],corruption=[0, 10, 10],max_patterns=[1, 30, 1])

    GeneralErrorstuff(filename="SimplicialHopfield1",HopfieldType="SimplicialHopfield",nums_neurons=[25],corruption=[0, 10, 10],max_patterns=[1, 30, 1])
    GeneralErrorstuff(filename="SimplicialHopfield2",HopfieldType="SimplicialHopfield",nums_neurons=[15],corruption=[0, 10, 10],max_patterns=[1, 30, 1])
    GeneralErrorstuff(filename="SimplicialHopfield3",HopfieldType="SimplicialHopfield",nums_neurons=[30],corruption=[0, 10, 10],max_patterns=[1, 30, 1])