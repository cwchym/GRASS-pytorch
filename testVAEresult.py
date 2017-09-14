from __future__ import division, print_function
import torch
from TrainVAE import myVAE
import draw3dOBB
from data_loader import get_loader
from torch.autograd import Variable
import numpy as np

def testDemo(index):
    index = index*20
    VAE = myVAE(200, 80, 8, 12, 3)
    data = get_loader('./data/trainingData_chair.mat', 1, False)
    d = list(enumerate(data))
    boxes = d[index][1]['boxes']
    symshapes = Variable(d[index][1]['symshapes'])
    treekids = d[index][1]['treekids']
    symparams = Variable(d[index][1]['symparams'])
    VAE.load_state_dict(torch.load('VAE.pkl'))

    boxes = boxes.numpy().squeeze()
    boxes = np.transpose(boxes).reshape([1,np.shape(boxes)[1],np.shape(boxes)[0]])
    #boxes = boxes.numpy()

    draw3dOBB.showGenshapes(boxes)

    num = 1
    tmpOut = VAE.testVAE(num,symshapes,treekids,symparams)
    tmpOut2,_,_,_,_,_,_ = VAE(symshapes,treekids,symparams)

    tmpOut2 = tmpOut2.data.numpy().reshape([1, tmpOut2.size(0), tmpOut2.size(1)])

    draw3dOBB.showGenshapes(tmpOut)
    draw3dOBB.showGenshapes(tmpOut2)

testDemo(5)