#Implementation of hopfield network
#General format idea adapted from pseudocode outline found here:
#https://www.geeksforgeeks.org/hopfield-neural-network/

import numpy as np

from scipy.special import softmax
#from scipy.special import logsumexp
import random

class Hopfield:
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):

        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.zeros((self.n,self.n)) #Connection matrix
        self.X = np.array(inputs)

        #w(ij) ​= ∑p​[s(i)​(p) * s(j)​(p)]
        for i in range(self.itemsLen):
            self.weights += np.outer(inputs[i], inputs[i])

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
    
class DAMDiscreteHopfield:
    #based on 'Dense Associative Memory for Pattern Recognition' paper

    #Initialisation function
    def __init__(self, inputs, rectified=True, power=2):
        self.n = len(inputs[0]) #no. of neurons
        self.N = len(inputs) # no. of patterns
        self.X = np.copy(inputs)
        self.rectified = rectified
        self.power = power
        
    
    #Update rule
    #Asynchronously flips all bits randomly
    #Keeps flipped bit if energy is lowered
    def predict(self, input, iterations = 5):

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

                current = self.energy(new_vals)

                if (current - prev) < 0:
                    prev = self.energy(new_vals)
                    vals[i] = new_vals[i]
                    noFlip = False
            if noFlip:
                break
            predicted.append(vals)
        return predicted
    
    #-∑F(state * x)
    def energy(self, state):
        x = self.X@state
        return -self.F(x, self.power).sum()
    
    #F (x) = {if x > 0, x^n, else 0}
    def F(self, x, n):
        if self.rectified:
            x[x < 0] = 0.
        return x**n

#Continuous Hopfield
#Based on:
#Hopfield Networks is All You Need
class ContinuousHopfield:
    #based on 'Dense Associative Memory for Pattern Recognition' paper

    #Initialisation function
    def __init__(self, inputs):
        self.n = len(inputs[0]) #no. of neurons
        self.M = np.linalg.norm(inputs[0])
        self.N = len(inputs) # no. of patterns
        self.X = np.copy(inputs)

        newX = np.copy(self.X)
        self.newX = np.array([newX[i]/np.mean(newX[i]) for i in range(len(newX))]) # original code
        #self.newX = np.array([newX[i]/newX[i] for i in range(len(newX))]) # removed mean
        #self.newX = newX
    
    #Update rule
    #X softmax(beta X^T ξ)
    def predict(self, input, iterations = 1, beta = 8):
        predicted = [np.copy(input)]
        #energy = self.energy(input, beta)

        for i in range(iterations):
            vals = softmax(beta * predicted[i] @ np.transpose(self.newX) ) @ self.X 
            #vals = softmax(beta * input @ np.transpose(self.X) ) @ self.X 
        
            #new_energy = self.energy(vals, beta)
            #if not new_energy < energy:
            #    break
            #print("ENERGY", new_energy, energy, new_energy< energy, 2 * self.M**2, self.energy(vals, beta) < 2 * self.M**2)
            
            if (vals == predicted[i]).all():
                break
            predicted.append(vals)
        return predicted
    
    # log(∑i[exp(βxi)])/β
    def LSE(self, beta, X):
        return np.log(np.sum([np.exp(beta*X[i]) for i in range(len(X))])) / beta
    
    #Energy rule
    # E = − lse(β, X^T ξ) + 0.5 * ξ^T ξ + log(N)/β + 0.5 * M^2   
    def energy(self, state, beta):
        lse = -np.log(np.sum([np.exp(beta * self.X[i] * state) for i in range(len(self.X))])) / beta
        x = lse + 0.5*np.transpose(state)@state + np.log(self.N)/beta + 0.5 * self.M**2
        return x


# To understand Simplicial Complex, consider the following example:
# if number of neurons = 6
# a 2-skeleton simplicial complex will contain the following pairwise and setwise connections
# between the neurons: 
#    [ 
#        [  
#           {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {3, 4}, {3, 5},
#           {3, 6}, {4, 5}, {4, 6}, {5, 6}
#        ],
#        [  
#            {1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 2, 6}, {1, 3, 4}, {1, 3, 5}, {1, 3, 6}, {1, 4, 5}, {1, 4, 6},
#            {1, 5, 6}, {2, 3, 4}, {2, 3, 5}, {2, 3, 6}, {2, 4, 5}, {2, 4, 6}, {2, 5, 6}, {3, 4, 5}, {3, 4, 6},
#            {3, 5, 6}, {4, 5, 6}
#        ]
#    ]
class SimplicialHopfield(Hopfield):
    def __init__(self, inputs, pairwise_connections=False):
        self.pairwise_connections = pairwise_connections
        if self.pairwise_connections:
            super().__init__(inputs)
        self.n = len(inputs[0])  # Number of neurons
        self.itemsLen = len(inputs)  # Number of patterns
        self.weights_k2 = np.zeros((self.n, self.n, self.n))  # Connection matrix
        self.X = np.array(inputs)  # Store the input patterns

        # Loop through each pattern and update the K2 weights
        #w(ijk) ​= ∑p​[s(i)​(p) * s(j)​(p) * s(k)(p)]
        for p in range(self.itemsLen):
            # Get the current pattern
            pattern = inputs[p]

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        if i != j != k:  # Ignore self-connections
                            self.weights_k2[i][j][k] += pattern[i] * pattern[j] * pattern[k]

        # Normalize the weights by the number of patterns
        self.weights_k2 = self.weights_k2 / self.itemsLen

    #Prediction function
    #yini ​​= ∑d[xi​ + ∑j​[yj * ​wji​]] where d ∈ K
    #yi = yini{1 if > =0, else 0}
    #Iterates until hits iteration count or energy minimized
    def predict(self, input, iterations, theta = 0.0):

        predicted = [np.copy(input)]

        s = self.energy(predicted[0])

        for i in range(0, iterations):
            if self.pairwise_connections:
                simplicial_update_sum = (self.weights @ predicted[i]) + (self.weights_k2 @ predicted[i]) 
            else:
                simplicial_update_sum = self.weights_k2 @ predicted[i]

            newVal = np.sign(simplicial_update_sum)[0]

            st = self.energy(newVal, theta = theta)
            if s == st:
                break
            s = st
            predicted.append(newVal)
        return predicted

    #E = -(1/2 * ∑i​∑j[​wij​ * vi * ​vj] + 1/3 *  ∑i​∑j∑k[​wijk​ * vi * ​vj * vk]) ​+ ∑i[​θi​ * vi] for {1, 2} ∈ K 
    def energy(self, state, theta=0.0):
        energy_sum = 0.0
        energy_sum_k2 = 0.0
        n = len(state)

        # Loop over all sets of neurons (i, j, k) to calculate the summation for weights and states for 2-skeleton simplex
        for i in range(n):
            for j in range(n):
                if self.pairwise_connections and i != j:
                    energy_sum += self.weights[i][j] * state[i] * state[j]
                for k in range(n):
                    if i != j != k:  # Exclude self-connections
                        energy_sum_k2 += self.weights_k2[i][j][k] * state[i] * state[j] * state[k]

        # Apply the simplicial hopfield energy rule
        print(energy_sum_k2)
        energy = ((-1/2) * energy_sum) + ((-1/3) * energy_sum_k2)

        # Sum over theta[i] * state[i]]
        t = [theta] * n
        theta_sum = 0.0
        for i in range(n):
            theta_sum += t[i] * state[i]
        
        # Get the final energy
        energy += theta_sum

        return energy
