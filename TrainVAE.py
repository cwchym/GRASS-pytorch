from __future__ import print_function, division
import math
import torch
import torch.nn as NN
import model
from torch.autograd import Variable

class myVAE(NN.Module):
    '''
        Train the entire network of VAE for Rvnn!
    '''
    def __init__(self,symshapes,treekids,symparams,hiddenSize,latentSize,symSize,boxSize,catSize):
        super(myVAE, self).__init__()
        self.shapes = symshapes
        self.treekids = treekids
        self.params = symparams
        self.hidden=hiddenSize
        self.latent=latentSize
        self.sym=symSize
        self.box=boxSize
        self.cat=catSize
        #define Rvnn encoder tree node
        self.encoder = model.RvnnEncoCell(hiddenSize,latentSize,symSize,boxSize)
        #define Rvnn decoder tree node
        self.decoder = model.RvnnDecoCell(hiddenSize,latentSize,symSize,boxSize,catSize)
        #define the middle layers between encoder and decoder
        self.tanh = NN.Tanh()
        self.ranen1 = NN.Linear(latentSize,hiddenSize)
        self.ranen2 = NN.Linear(hiddenSize,latentSize*2)
        self.rande2 = NN.Linear(latentSize,hiddenSize)
        self.rande1 = NN.Linear(hiddenSize,latentSize)

        for i, m in enumerate(self.modules()):
            if(i != 0):
                if(isinstance(m,NN.Linear)):
                    r = math.sqrt(6/(self.hidden+self.hidden+1))
                    m.weight.data = torch.rand(m.weight.data.size())*2*r-r

    def VAEencoder(self):
        nodeNum = self.treekids.size(1)
        hOut = Variable(torch.DoubleTensor(nodeNum,self.latent).zero_(), requires_grad = True)
        leafNum = self.shapes.size(2)
        for i in range(nodeNum):
            if(self.treekids[0,i,:][0] == 0):
                nodeType = 0
                input1 = self.shapes[0,:,i]
                input2 = None
            if(self.treekids[0,i,:][0] != 0 and self.treekids[0,i,:][2] == 0):
                nodeType = 1
                if(i > leafNum):
                    input1 = hOut[self.treekids[0,i,0],:]
                    input2 = hOut[self.treekids[0,i,1],:]
            elif(self.treekids[0,i,:][0] != 0 and self.treekids[0,i,:][2] == 1):
                nodeType = 2
                if(i > leafNum):
                    input1 = hOut[self.treekids[0,i,0],:]
                    input2 = self.params[0, self.treekids[0,i,0], :]

            hOut[i,:] = self.encoder(nodeType,input1,input2)

        return hOut[nodeNum-1,:]

    def VAEdecoder(self,rd1):
        nodeNum = self.treekids.size(1); leafNum = self.shapes.size(2)
        paramOut = Variable(torch.DoubleTensor(self.params.size(1),self.params.size(2)).zero_(), requires_grad = True)
        paramGT = Variable(torch.DoubleTensor(self.params.size(1),self.params.size(2)).zero_(), requires_grad = True)
        NClrOut = Variable(torch.DoubleTensor(nodeNum,self.cat).zero_(), requires_grad = True)
        NClrGroundTruth = Variable(torch.DoubleTensor(nodeNum,1).zero_(), requires_grad = True)
        hmidOut = Variable(torch.DoubleTensor(nodeNum,self.latent).zero_(), requires_grad = True)
        hOut = Variable(torch.DoubleTensor(leafNum,self.box).zero_(), requires_grad = True)
        hmidOut[nodeNum - 1, :] = rd1

        for i in reversed(range(nodeNum)):
            nodeType = 0
            if (self.treekids[0, i, :][0] == 0):
                nodeType = 0
                NClrGroundTruth[i, :] = Variable(torch.ByteTensor([0]))
            elif(self.treekids[0,i,:][0] != 0 and self.treekids[0,i,:][2] == 0):
                #Adjacent Node
                nodeType = 1
                NClrGroundTruth[i, :] = Variable(torch.ByteTensor([1]))
            elif(self.treekids[0,i,:][0] != 0 and self.treekids[0,i,:][2] == 1):
                #Symmetry Node
                nodeType = 2
                NClrGroundTruth[i, :] = Variable(torch.ByteTensor([2]))
                paramGT[i,:] = self.params[0,self.treekids[0, i, 0], :]

            input1 = hmidOut[i,:]
            tmpOut = self.decoder(nodeType,input1)

            if(self.treekids[0,i,:][0] == 0):
                hOut[i, :] = tmpOut.narrow(1, 0, self.box)
                NClrOut[i, :] = tmpOut.narrow(1, self.box, self.box+self.cat)
            elif(self.treekids[0,i,:][0] != 0 and self.treekids[0,i,:][2] == 0):
                id1 = self.treekids[0,i,0]
                id2 = self.treekids[0,i,1]
                hmidOut[id1, :] = tmpOut.narrow(1, 0, self.latent)
                hmidOut[id2, :] = tmpOut.narrow(1, self.latent, self.latent+self.latent)
                NClrOut[i, :] = tmpOut.narrow(1, self.latent+self.latent, self.latent+self.latent+self.cat)
            elif(self.treekids[0, i, :][0] != 0 and self.treekids[0, i, :][2] == 1):
                id1 = self.treekids[0, i, 0]
                hmidOut[id1, :] = tmpOut.narrow(1, 0, self.latent)
                paramOut[id1, :] = tmpOut.narrow(1, self.latent, self.latent + self.sym)
                NClrOut[i, :] = tmpOut.narrow(1, self.latent + self.sym, self.latent + self.sym + self.cat)

        return hOut, paramOut, paramGT, NClrOut, NClrGroundTruth

    def forward(self):
        encOutput = self.VAEencoder()
        re1 = self.tanh(self.ranen1(encOutput))
        re2 = self.ranen2(re1)

        mu = re2.narrow(1, 0, self.latent)
        logvar = re2.narrow(1, self.latent, self.latent*2)

        sig = torch.exp(logvar/2)
        eps = Variable(torch.randn(mu.size()), requires_grad = True)
        sample_z = mu + sig * eps

        rd2 = self.tanh(self.rande2(sample_z))
        rd1 = self.tanh(self.rande1(rd2))

        Out, paramOut, paramGT, NClrOut, NClrGroudTruth = self.VAEdecoder(rd1)

        return Out, paramOut, paramGT, NClrOut, NClrGroudTruth, mu, sig