import cv2
import numpy as np
from model import MyModel
import torchvision
import torch

#-----------------------------------------------------
device = torch.device("cpu")
model = MyModel()
model = model.to(device)
model.train(True)
#----------------------------------------------------
transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0),(1))
            # torchvision.transforms.RandomHorizontalFILIP(),
])
#___________________________________________________________
torch.save(model.state_dict(),"fashion_mnist")
#-----------------------------------------------------------
model.eval()


img = cv2.imread('image-22.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img,(28,28))
tensor = transform(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()

output = np.argmax(preds)
print(output)