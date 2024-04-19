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
    
#Continuous Hopfield
#Based on:
#Hopfield Networks is All You Need
class ContinuousHopfield(nn.Module):
    #Initialisation function
    def __init__(self, inputs, beta = 8):
        super().__init__()
        self.X = nn.Parameter(torch.Tensor(np.copy(inputs)))
        self.beta = beta
        newX = np.copy(inputs)
        #self.newX = nn.Parameter(torch.Tensor(np.array([newX[i]/np.mean(newX[i]) for i in range(len(newX))])))
        self.newX = torch.Tensor(np.array([newX[i]/np.mean(newX[i]) for i in range(len(newX))]))
        
    #Update rule
    #X softmax(beta X^T ξ)
    def forward(self, input):
        #predicted = [np.copy(input)]
        #vals = softmax(self.beta * input @ np.transpose(self.newX) ) @ self.X 
        vals = torch.matmul(torch.Tensor(softmax(self.beta * torch.matmul(input, torch.transpose(self.newX, 0, 1)))), self.X) 
        #vals = softmax(beta * input @ np.transpose(self.X) ) @ self.X 
        return vals
    

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data



class myModel: 
    def __init__(self, inputData, learningRate = 0.001, momentum = 0.9, loss_fn = nn.CrossEntropyLoss()):
        self.net = nn.Sequential(
            ContinuousHopfield(inputData),
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

def prepro(img):
    #print(type(img))
    #print(img)
    return torch.flatten(img/255).float()
    #return torch.from_numpy(img/255).float()


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