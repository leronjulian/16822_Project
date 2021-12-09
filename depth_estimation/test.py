import numpy as np
from numpy.core.numeric import base_repr 
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
from dataLoaders_test import DatasetFromFolder

dir_paths = ['checkboard_pairs/', 'floor_pairs/', 'coolGlass/']
out_paths = ['checkboard_pairs_depth/', 'floor_pairs_depth/', 'coolGlass_depth/']

choice = 2
directoryName = dir_paths[choice]
outDirName = out_paths[choice]

# base_dir = "./data/test_imgs/pairs/"
base_dir = "./data/test_imgs/synthetic_pairs/"


FRAME_PATH =  base_dir + directoryName
OUT_PATH = base_dir + outDirName

# trainLoader = DataLoader(DatasetFromFolder(FRAME_PATH), batch_size = 32, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

iterNum = 170
checkpoint_location = './Iteration' + str(iterNum) + '.pth'
checkpoint = torch.load(checkpoint_location)

net = UNet(n_channels=6, n_classes=1, bilinear=True)
net.load_state_dict(checkpoint)
net = net.to(device)

for parameter in net.parameters():
        parameter.requires_grad = False
net.eval()

for i in range(39):
        indx = str(i)
        if i < 10:
                indx = '0' + indx
        fname1 = FRAME_PATH + indx + '_object.png'
        fname2 = FRAME_PATH + indx + '_background.png'


        img1 = plt.imread(fname1)[:,:,:3]
        img1 = cv2.resize(img1, dsize=(img1.shape[1]/4, img1.shape[0]/4), interpolation=cv2.INTER_CUBIC)


        img2 = plt.imread(fname2)[:,:,:3]
        img2 = cv2.resize(img2, dsize=(img2.shape[1]/4, img2.shape[0]/4), interpolation=cv2.INTER_CUBIC)

        img2 = img2/255

        inputs = np.moveaxis(img1, 2, 0)
        inputs = torch.from_numpy(inputs)
        

        inputs_noBK = np.moveaxis(img2, 2, 0)
        inputs_noBK = torch.from_numpy(inputs_noBK)

        inputs = torch.cat((inputs_noBK, inputs), 0) 
        inputs = torch.unsqueeze(inputs, 0)

        inputs = inputs.to(device)

        output = net(inputs)

        img = output.cpu().detach().numpy()[0,:,:,:]
        img = np.squeeze(img, 0)
        plt.imsave(base_dir + outDirName + indx + '_depth.png', img, cmap='gray')

