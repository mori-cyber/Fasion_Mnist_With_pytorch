# train_model
# Hyper parameter
import torch
import torchvision
from model import MyModel

#------------------------------------------------------------------------------------------
def calc_acc(preds,labels):
  _,preds_max = torch.max(preds, 1)
  acc = torch.sum(preds_max==labels.data,dtype = torch.float64)/len(preds)
  return acc

#--------------------------------------------------------------------------------------------------------
device =torch.device("cpu")
# device = torch.device("cuda")
model = MyModel()
model = model.to(device)
model.train(True)
#--------------------------------------------------------------------------------------------------------
#Hyper Parameter
batch_size=64
epoch=20
lr=0.001
#---------------------------------------------------------------------------------------------------------
#Data Preparing
transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0),(1))
            # torchvision.transforms.RandomHorizontalFILIP(),
])

dataset = torchvision.datasets.FashionMNIST("./dataset",train =True,download=True, transform = transform )
train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#---------------------------------------------------------------------------------------------------------
#compile
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_function = torch.nn.CrossEntropyLoss()
#----------------------------------------------------------------------------------------------------------
#train
for epochs in range(epoch):
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 1-forwarding
        preds = model(images)
        # 2- backwarding
        loss = loss_function(preds, labels)
        loss.backward()
        # 3- update
        optimizer.step()
        train_acc += calc_acc(preds, labels)
        train_loss += loss
    total_loss = train_loss / len(train_data_loader)
    total_acc = train_acc / len(train_data_loader)

    print(f"Epoch:{epochs}: Loss:{total_loss}")
    print(f"Acc:{epochs}: acc:{total_acc}")
    print("-------------------------------------")