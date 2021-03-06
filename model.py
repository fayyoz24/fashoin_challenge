from torch import nn, optim
import matplotlib.pyplot as plt
from helper_functions import acc


# class Classify(nn.Module):
#     def __init__(self,inputdimension,outdim):
#         super(Classify,self).__init__()
#         self.in_lier = nn.Linear(inputdimension,10)
#         self.hidden0 = nn.Linear(10,800)
#         self.hidden1 = nn.Linear(800, 400)
#         self.hidden2 = nn.Linear(400, 200)
#         self.hidden3 = nn.Linear(200, 100)
#         self.hidden4 = nn.Linear(100, 50)
#         self.output  = nn.Linear(50,outdim)
#         self.sigmoid    = nn.ReLU()

#     def forward(self, x):
#         input = self.in_lier(x)
#         ac1 = self.sigmoid(input)
#         input1 = self.hidden0(ac1)
#         ac2 = self.sigmoid(input1)
#         input3 = self.hidden1(ac2)
#         ac3 = self.sigmoid(input3)
#         input4 = self.hidden2(ac3)
#         ac4 = self.sigmoid(input4)
#         input5 = self.hidden3(ac4)
#         ac5 = self.sigmoid(input5)
#         input6 = self.hidden4(ac5)
#         ac6 = self.sigmoid(input6)
#         output = self.output(ac6)
#         return output


# class Classify(nn.Module):
#     def __init__(self,inputdimension,outdim):
#         super(Classify,self).__init__()
#         self.in_lier = nn.Linear(inputdimension,1600)
#         self.hidden0 = nn.Linear(1600,400)
#         self.hidden1 = nn.Linear(400,100)
#         self.output  = nn.Linear(100,outdim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         input = self.in_lier(x)
#         ac1 = self.sigmoid(input)
#         input2 = self.hidden0(ac1)
#         ac2 = self.sigmoid(input2)
#         input3 = self.hidden1(ac2)
#         ac3 = self.sigmoid(input3)
#         output = self.output(ac3)
#         return output

class Classify(nn.Module):
    def __init__(self,inputdimension,outdim):
        super(Classify,self).__init__()
        self.in_lier = nn.Linear(inputdimension,1600)
        self.hidden0 = nn.Linear(1600,400)
        self.hidden1 = nn.Linear(400,100)
        self.output  = nn.Linear(100,outdim)
        # self.log_softmax = nn.LogSoftmax(dim = 1)
        self.relu = nn.ReLU()
        # self.drop = nn.Dropout(0.2)

    def forward(self, x):
        input = self.in_lier(x)
        # ac1 = self.log_softmax(input)
        ac1 = self.relu(input)
        # ac1 = self.drop(ac1)
        input2 = self.hidden0(ac1)
        # ac2 = self.log_softmax(input)
        ac2 = self.relu(input2)
        # ac2 = self.drop(ac2)
        input3 = self.hidden1(ac2)
        # ac3 = self.log_softmax(input)
        ac3 = self.relu(input3)
        # ac3 = self.drop(ac3)
        output = self.output(ac3)
        return output