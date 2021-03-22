import torch
import torch.nn as nn 


class Net(nn.Module):

    def __init__(self,input_size, layers_specs, cuda):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_specs = layers_specs
        self.input_size =  input_size
        current_dim = input_size
        for layer in self.layers_specs:
            self.layers.append(nn.Linear(current_dim, layer["neurons"]))
            current_dim = layer["neurons"]
        if cuda: 
            self.cuda()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activate_input(self.layers[i](x), self.layers_specs[i]["activation"])
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
