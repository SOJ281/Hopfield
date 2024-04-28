#Hopfield and DAM class

import numpy as np
from matplotlib import pyplot as plt
from hopfield import *

import numpy as np
from os import listdir


from matplotlib import pyplot as plt

import torch.nn as nn
import torch
from torch import nn, optim
from torch import optim

import csv
from tqdm import tqdm

class Hopfield(nn.Module):
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):
        super().__init__()
        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.empty((self.n,self.n)) #Connection matrix
        self.X = np.array(inputs)

        #w(ij) ​= ∑p​[s(i)​(p) * s(j)​(p)]
        for i in range(self.itemsLen):
            self.weights += np.outer(inputs[i], inputs[i])

        for i in range(self.n):
            self.weights[i][i] = 0

        # notsure if this step is necessary
        self.weights = nn.Parameter(torch.Tensor(self.weights/self.itemsLen))

    #Prediction function
    #yini ​​= xi​ + ∑j​[yj * ​wji​]
    #yi = yini{1 if >= 0, else 0}
    #Iterates until hits iteration count or energy minimized
    #def predict(self, input, iterations, theta = 0.0):
    def forward(self, input):
        predicted = torch.sign(self.weights @ input)
        return predicted

    #E = 0.5 * ∑i​∑j[​wij​ * vi * ​vj] ​+ ∑i[​θi​ * vi]
    def energy(self, state, theta = 0.0):
        return -0.5 * state @ self.weights @ state + np.sum(state*theta)
    
class DAMDiscreteHopfield(nn.Module):
    #based on 'Dense Associative Memory for Pattern Recognition' paper
    #Initialisation function
    def __init__(self, inputs):
        super().__init__()
        self.n = len(inputs[0]) #no. of neurons
        self.N = len(inputs) # no. of patterns
        self.Weights = nn.Parameter(torch.Tensor(np.copy(inputs)))
        
    #Update rule
    def forward(self, input):
        valList = np.arange(0, self.n)
        random.shuffle(valList)
        vals = input.detach().clone()
        prev = self.energy(vals)
        for i in valList:
            new_vals = vals.detach().clone()
            new_vals[i] *= -1
            current = self.energy(new_vals)

            if current < prev:
                prev = current
                vals[i] = new_vals[i]

        return vals
    
    #-∑F(state * x)
    def energy(self, state):
        x = torch.matmul(self.Weights, state)
        return -self.F(x, 2).sum()
    
    #F (x) = {if x > 0, x^n, else 0}
    def F(self, x, n):
        x[x < 0] = 0.
        return x**n
    
class DAM(nn.Module):
    #based on 'Dense Associative Memory for Pattern Recognition' paper
    #Initialisation function
    def __init__(self, inputs):
        super().__init__()
        self.n = len(inputs[0]) #no. of neurons
        self.K = len(inputs) # no. of patterns
        self.Weights = nn.Parameter(torch.Tensor(np.copy(inputs)))
        
    #Update rule
    def forward(self, input):
        valList = np.arange(0, self.n)
        #random.shuffle(valList)
        vals = input.detach().clone()
        #prev = self.energy(vals)
        for i in valList:
            neg_vals = vals.detach().clone()
            neg_vals[i] *= -1
            neg_vals = self.energy(neg_vals)

            pos_vals = vals.detach().clone()
            pos_vals = self.energy(pos_vals)


            #vals[i] = torch.sign(pos_vals - neg_vals)
            vals[i] = torch.tanh(pos_vals - neg_vals)
        #print("SHED")
        return vals
    
    #-∑F(state * x)
    def energy(self, state):
        x = torch.matmul(self.Weights, state)
        return -self.F(x, 2).sum()
    
    #F (x) = {if x > 0, x^n, else 0}
    def F(self, x, n):
        #print(x)
        x[x < 0] = 0.
        return x**n

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data



class myModel: 
    def __init__(self, inputData, learningRate = 0.001, momentum = 0.9, loss_fn = nn.CrossEntropyLoss()):
        self.net = nn.Sequential(
            #Hopfield(inputData),
            #DAMDiscreteHopfield(inputData),
            DAM(inputData),
            #nn.ReLU(),
            nn.Linear(len(inputData[0]), 10)
        )
            
        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.net.parameters(), lr = learningRate, momentum = momentum)

    def testResults(self, testData, testDataLabels):
        correct = 0
        running_loss = 0
        with torch.no_grad():
            for data in range(len(testData)):
                inputs = testData[data]
                labels = testDataLabels[data] 

                outputs = self.net(inputs)

                # accumulate loss
                running_loss += self.loss_fn(outputs, labels)

                # accumulate data for accuracy
                _, predicted = torch.max(outputs.data, 0)
                correct += (predicted == labels)

        return correct/len(testData), running_loss/len(testData)
    
    def predict(self, input):
        with torch.no_grad():
            output = self.net(input.float())
        return output

    #training loop
    def eval(self, nepochs, trainingData, trainingDataLabels, testData, testDataLabels):
        trainingAcc = []
        trainingLoss = []
        testAcc = []
        testLoss = []
        best = 0
        for epoch in range(nepochs):  # loop over the dataset multiple times
            correct = 0          
            running_loss = 0.0                 
            for data in tqdm(range(len(trainingData))):
                inputs = trainingData[data]
                labels = trainingDataLabels[data] 

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward, backward, and update parameters
                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # accumulate loss
                running_loss += loss.item()

                # accumulate data for accuracy
                _, predicted = torch.max(outputs.data, 0)
                correct += (predicted == labels)

            trainingAcc.append(correct/len(trainingData))
            trainingLoss.append(running_loss/len(trainingData))
            print("Epoch:", epoch)
            print("Train Accuracy:",correct/len(trainingData), "Train Loss:",running_loss/len(trainingData))
            testa, testl = self.testResults(testData, testDataLabels)
            testAcc.append(testa)
            testLoss.append(testl)
            print("Test Accuracy:", testa, "Test Loss:", testl)
            print("===========================================================")

            if testAcc[best] < testa:
                best = epoch
                
        return trainingAcc, trainingLoss, testAcc, testLoss, best
    
from torchvision import datasets, transforms
train_data = datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)

test_data = datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

def prepro(img):
    flatty = torch.flatten(torch.where(img > torch.mean(img.float()), 1, -1))
    return flatty.float()

trainingData = train_data.data[:1000]
#trainingLabels = torch.from_numpy(np.array([int(i[0]) for i in trainingData])).type(torch.LongTensor)
trainingLabels = train_data.train_labels
trainingData = [prepro(i) for i in trainingData]

valsTest = test_data.data[:500]
testLabels = test_data.test_labels
#testLabels = torch.from_numpy(np.array([int(i[0]) for i in valsTest])).type(torch.LongTensor)
testData = [prepro(i) for i in valsTest]

epochs = 5
"""
learningRates=[0.001, 0.01]
momentum=[0.2, 0.5, 0.9]
nesterov = [True, False]

import itertools
for lr, mom in list(itertools.product(learningRates, momentum)):
    print("Training")
    mlp = myModel(trainingData[:10], learningRate = lr, momentum = mom, loss_fn = nn.CrossEntropyLoss())
    trainingAcc, trainingLoss, testAcc, testLoss, best = mlp.eval(epochs, trainingData, trainingLabels, testData, testLabels)
    print("Best:", (lr, mom), "Values:")
    print("Training:", trainingAcc[best], trainingLoss[best])
    print("Test:", testAcc[best], testLoss[best])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
"""
def draw_weights(synapses, Kx, Ky, err_tr, err_test):
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    plt.sca(ax1)
    yy=0
    HM=np.zeros((28*Kx,28*Ky))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    cbar=fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    cbar.ax.tick_params(labelsize=30) 
    
    plt.sca(ax2)
    plt.ylim((0,100))
    plt.xlim((0,len(err_tr)+1))
    ax2.plot(np.arange(1, len(err_tr)+1, 1), err_tr, color='b', linewidth=4)
    ax2.plot(np.arange(1, len(err_test)+1, 1), err_test, color='r',linewidth=4)
    ax2.set_xlabel('Number of epochs', size=30)
    ax2.set_ylabel('Training and test error, %', size=30)
    ax2.tick_params(labelsize=30)

    plt.tight_layout()
    fig.canvas.draw()

print("Training")
mlp = myModel(trainingData[100:], learningRate = 0.001, momentum = 0.5, loss_fn = nn.CrossEntropyLoss())
trainingAcc, trainingLoss, testAcc, testLoss, best = mlp.eval(epochs, trainingData, trainingLabels, testData, testLabels)
print("Training:", trainingAcc[best], trainingLoss[best])
print("Test:", testAcc[best], testLoss[best])
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

epochs = [i+1 for i in range(epochs)]

plt.plot(epochs, trainingAcc, label='training', color='blue', linewidth=3)
plt.plot(epochs, testAcc, label='test', color='red', linewidth=3)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, trainingLoss, label='training', color='blue', linewidth=3)
plt.plot(epochs, testLoss, label='test', color='red', linewidth=3)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title('Loss')
plt.legend()
plt.show()

for i in range(len(trainingData)):
    longest  = 1
    _, predicted = torch.max(mlp.predict(trainingData[i]).data, 0)
    print("Predicted:", predicted)
    print("Actual:", trainingLabels[i])
    fig, axarr = plt.subplots(1, 1, figsize=(5, 5))
    axarr.set_title('Originals')
    axarr.imshow(reshape(trainingData[i]))
    axarr.axis('off')

    plt.tight_layout()
    plt.show()

    if input("Hit y to stop, or anything else to keep going") == 'y':
        break