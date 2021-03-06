import torch
import torch.nn as nn 
#import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self,input_size, layers_specs, cuda):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_specs = layers_specs
        self.input_size =  input_size
        self.activation_output = nn.LogSoftmax(dim=1)
        current_dim = input_size
        for layer in self.layers_specs:
            self.layers.append(nn.Linear(current_dim, layer["neurons"]))
            current_dim = layer["neurons"]
        if cuda: 
            self.cuda()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activate_input(self.layers[i](x), self.layers_specs[i]["activation"])

        #print("summen skal vaere 1 ", torch.sum(x))
        if len(list(x.size())) == 2:
            x = self.softmax_1(x)
        elif len(list(x.size())) == 3:
            x = self.softmax_2(x)


        return x

    def activate_input(self, x, activation_name):
        if activation_name == "relu":
            return torch.relu(x)
        elif activation_name == "tanh":
            return torch.tanh(x)
        elif activation_name == "sigmoid":
            return torch.sigmoid(x)
        elif activation_name == "selu":
            return torch.selu(x)
        elif activation_name == "linear":
            return x
        elif activation_name == "softmax":
            #print("output ", x)
            #print("activated ",  self.softmax(x))
            #print("sum", torch.sum(self.softmax(x)))
            return self.softmax_1(x)
