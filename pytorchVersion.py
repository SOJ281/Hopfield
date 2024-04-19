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
    

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data



class myModel: 
    def __init__(self, inputData, learningRate = 0.001, momentum = 0.9, loss_fn = nn.CrossEntropyLoss()):
        self.net = nn.Sequential(
            Hopfield(inputData),
            #DAMDiscreteHopfield(inputData),
            nn.ReLU(),
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
        for epoch in range(nepochs):  # loop over the dataset multiple times
            correct = 0          
            running_loss = 0.0                 
            for data in range(len(trainingData)):
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
            testacc, testloss = self.testResults(testData, testDataLabels)
            testAcc.append(testacc)
            testLoss.append(testloss)
            print("Test Accuracy:", testacc, "Test Loss:", testloss)
            print("===========================================================")
        return trainingAcc, trainingLoss, testAcc, testLoss
    
from torchvision import datasets, transforms
train_data = datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)

test_data = datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

#print(train_data.data)
#print(train_data.train_labels)
#vals = []
#with open("archive/mnist_train.csv") as csvfile:
#    reader = csv.reader(csvfile) # change contents to floats
#    for row in reader: # each row is a list
#        try:
#          vals.append(row)
#        except:
#          pass

#def prepro(img):
    #print(type(img))
    #print(img)
#    return torch.flatten(img/255).float()
    #return torch.from_numpy(img/255).float()

def prepro(img):
    flatty = torch.flatten(torch.where(img > torch.mean(img.float()), 1, -1))
    return flatty.float()

trainingData = train_data.data[:5000]
#trainingLabels = torch.from_numpy(np.array([int(i[0]) for i in trainingData])).type(torch.LongTensor)
trainingLabels = train_data.train_labels
trainingData = [prepro(i) for i in trainingData]
#valsTest = []
#with open("archive/mnist_test.csv") as csvfile:
#    reader = csv.reader(csvfile) # change contents to floats
#    for row in reader: # each row is a list
#        try:
#          valsTest.append(row)
#        except:
#          pass

valsTest = test_data.data
testLabels = test_data.test_labels
#testLabels = torch.from_numpy(np.array([int(i[0]) for i in valsTest])).type(torch.LongTensor)
testData = [prepro(i) for i in valsTest]

epochs = 30
#print("SHED")
mlp = myModel(trainingData, learningRate = 0.001, momentum = 0.9, loss_fn = nn.CrossEntropyLoss())
print("Training")
trainingAcc, trainingLoss, testAcc, testLoss = mlp.eval(epochs, trainingData, trainingLabels, testData, testLabels)


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