
import torch
import torchvision


class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # input shape 32*28*28
        x = x.reshape((x.shape[0], 784))
        # output shape: 32*784
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.dropout(x, 0.2, train=True)
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x

