import numpy as np 
from unet_model import UNet
from skimage import color
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
from os import walk
import natsort 
from os import walk
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import os
import cv2
import os.path
import sys
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import os
# import cv2
import os.path
import sys
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from torchsummary import summary
import scipy.misc as m
from skimage.transform import resize
import imageio
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import os
import os.path
import sys
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
from os import walk
import natsort 
import h5py
from dataLoaders import DatasetFromFolder

FRAME_PATH = "./data/imgs/"
NO_BK_FRAME_PATH = "./data/bkgd/"
TARGET_PATH = "./data/depth/"
trainLoader = DataLoader(DatasetFromFolder(FRAME_PATH, NO_BK_FRAME_PATH, TARGET_PATH), batch_size = 32, shuffle=True)


# Use like 300 images

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
EPOCHS = 400


def custom_loss_function(output, target):
    di = target - output
    n = (target.shape[0] * target.shape[1])
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()



def run_train(model, x, y, lossfun, optimizer):
    target = y
    model = model.train()

    output = model(x)
    img = output.cpu().detach().numpy()[0,:,:,:]
    img = np.squeeze(img, 0)
    plt.imsave('./output.jpg', img)
  
    optimizer.zero_grad()

    loss = lossfun(output, y)

    # loss = nn.MSELoss(output, y)

    loss.backward()
    optimizer.step()
    
    return loss.item()




# Unet 
net = UNet(n_channels=6, n_classes=1, bilinear=True)

# net = torch.nn.DataParallel(net, device_ids=[0, 3]) 
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr = 0.01)



# Training Part ...
num_images = 0
for epoch in tqdm(range(EPOCHS), position = 0, leave = True):
    print('Starting Epoch...', epoch + 1)
    
    trainLossCount = 0
    num_images = 0
    for i, data in enumerate(trainLoader):
        # Training
        inputs = Variable(data[0]).to(device) # The input data
        target = Variable(data[1]).float().to(device)
        
        
        num_images += inputs.size(0)
        
        # Trains model
        trainingLoss = run_train(net, inputs, target, custom_loss_function, optimizer) 
        trainLossCount += trainingLoss
        
        
    trainLossCount /= (i + 1)

    print('Training Loss...')
    print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch + 1, i + 1, len(trainLoader), trainLossCount))
    if epoch % 10 == 0:
        PATH =  './Iteration' + str(epoch) + ".pth"
        torch.save(net.state_dict(), PATH)



print('Training Complete...')

# Saves UNET Network
PATH =  './Final_Model.pth'

torch.save(net.state_dict(), PATH)

print('Saved model to -> ' + PATH)


