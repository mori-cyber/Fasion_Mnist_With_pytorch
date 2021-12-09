


import torch
import torchvision
from model import MyModel

def calc_acc(preds,labels):
  _,preds_max = torch.max(preds, 1)
  acc = torch.sum(preds_max==labels.data,dtype = torch.float64)/len(preds)
  return acc
#-------------------------------------------------
#Hyper parameter
batch_size=64
epoch=20
lr=0.001
#------------------------------------------------
device =torch.device("cpu")
model = MyModel()
model = model.to(device)
model.train(True)
#-------------------------------------------------
transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0), (1)),
])

test_set = torchvision.datasets.FashionMNIST('./test_data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

model.load_state_dict(torch.load('fashion_mnist'))
model.eval()

test_acc=0.0
for img, label in test_loader:

    img = img.to(device)
    label = label.to(device)

    pred = model(img)
    test_acc += calc_acc(pred, label)

total_acc = test_acc / len(test_loader)
print(f"test accuracy: {total_acc}")