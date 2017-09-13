from __future__ import division, print_function
import torch
from TrainVAE import myVAE
import draw3dOBB

def testDemo():
    VAE = myVAE(200, 80, 8, 12, 3)
    VAE.load_state_dict(torch.load('VAE.pkl'))

    num = 1
    tmpOut = VAE.testVAE(num)

    draw3dOBB.showGenshapes(tmpOut)

testDemo()