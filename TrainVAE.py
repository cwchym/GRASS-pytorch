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
    def __init__(self, hiddenSize,latentSize,symSize,boxSize,catSize):
        super(myVAE, self).__init__()
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

    def VAEencoder(self,symshapes,treekids,symparams):
        nodeNum = treekids.size(1)
        hOut = list(Variable(torch.FloatTensor(1 ,self.latent).zero_(), requires_grad = True) for i in range(nodeNum))
        leafNum = symshapes.size(2)
        for i in range(nodeNum):
            if(treekids[0, i, :][0] == 0):
                nodeType = 0
                input1 = symshapes[0, :, i].contiguous().view(1, -1).float()
                input2 = None
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] == 0):
                nodeType = 1
                if(i >= leafNum):
                    input1 = hOut[treekids[0, i, 0]-1]
                    input2 = hOut[treekids[0, i, 1]-1]
            elif(treekids[0,i,:][0] != 0 and treekids[0,i,:][2] == 1):
                nodeType = 2
                if(i >= leafNum):
                    input1 = hOut[treekids[0,i,0]-1]
                    input2 = symparams[:, treekids[0, i, 0] - 1, :].float()

            hOut[i] = self.encoder(nodeType,input1,input2)

        return hOut[nodeNum-1]


    def concate(self, input1):
        myOut = input1[0]
        for i in range(1, len(input1)):
            tmpOut = input1[i]
            myOut = torch.cat((myOut, tmpOut), 0)

        return myOut

    def VAEdecoder(self, symshapes, treekids, symparams, rd1):
        nodeNum = treekids.size(1); leafNum = symshapes.size(2)
        paramOut = list(Variable(torch.FloatTensor(1, symparams.size(2)).zero_(), requires_grad = True) for i in range(symparams.size(1)))
        paramGT = list(Variable(torch.FloatTensor(1, symparams.size(2)).zero_(), requires_grad = True) for i in range(symparams.size(1)))
        NClrOut = list(Variable(torch.FloatTensor(1, self.cat).zero_(), requires_grad = True) for i in range(nodeNum))
        NClrGroundTruth = list(Variable(torch.FloatTensor(1, self.cat).zero_(), requires_grad = True) for i in range(nodeNum))
        hmidOut = list(Variable(torch.FloatTensor(1, self.latent).zero_(), requires_grad = True) for i in range(nodeNum))
        hOut = list(Variable(torch.FloatTensor(1, self.box).zero_(), requires_grad = True) for i in range(leafNum))
        hmidOut[nodeNum - 1] = rd1

        for i in reversed(range(nodeNum)):
            nodeType = 0
            if (treekids[0, i, :][0] == 0):
                nodeType = 0
                # NClrGroundTruth[i] = Variable(torch.LongTensor([0]))
                NClrGroundTruth[i] = Variable(torch.FloatTensor([[1, 0, 0]]))
            elif(treekids[0, i, :][0] != 0 and treekids[0, i,:][2] == 0):
                #Adjacent Node
                nodeType = 1
                # NClrGroundTruth[i] = Variable(torch.LongTensor([1]))
                NClrGroundTruth[i] = Variable(torch.FloatTensor([[0, 1, 0]]))
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] == 1):
                #Symmetry Node
                nodeType = 2
                # NClrGroundTruth[i] = Variable(torch.LongTensor([2]))
                NClrGroundTruth[i] = Variable(torch.FloatTensor([[0, 0, 1]]))
                paramGT[treekids[0, i, 0]-1] = symparams[:,treekids[0, i, 0]-1, :].float()

            input1 = hmidOut[i]
            tmpOut = self.decoder(nodeType,input1)

            if(treekids[0, i, :][0] == 0):
                hOut[i] = tmpOut.narrow(1, 0, self.box)
                NClrOut[i] = tmpOut.narrow(1, self.box, self.cat)
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] == 0):
                id1 = treekids[0, i, 0]
                id2 = treekids[0, i, 1]
                hmidOut[id1-1] = tmpOut.narrow(1, 0, self.latent)
                hmidOut[id2-1] = tmpOut.narrow(1, self.latent, self.latent)
                NClrOut[i] = tmpOut.narrow(1, self.latent+self.latent, self.cat)
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] != 0):
                id1 = treekids[0, i, 0]
                hmidOut[id1-1] = tmpOut.narrow(1, 0, self.latent)
                paramOut[id1-1] = tmpOut.narrow(1, self.latent, self.sym)
                NClrOut[i] = tmpOut.narrow(1, self.latent + self.sym, self.cat)

        hOut = self.concate(hOut)
        paramOut = self.concate(paramOut)
        paramGT = self.concate(paramGT)
        NClrOut = self.concate(NClrOut)
        NClrGroundTruth = self.concate(NClrGroundTruth)
        return hOut, paramOut, paramGT, NClrOut, NClrGroundTruth

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, symshapes, treekids, symparams):
        encOutput = self.VAEencoder(symshapes, treekids, symparams)
        re1 = self.tanh(self.ranen1(encOutput))
        re2 = self.ranen2(re1)

        mu = re2.narrow(1, 0, self.latent)
        logvar = re2.narrow(1, self.latent, self.latent*2)

        sample_z = self.reparameterize(mu, logvar)

        rd2 = self.tanh(self.rande2(sample_z))
        rd1 = self.tanh(self.rande1(rd2))

        myOut, paramOut, paramGT, NClrOut, NClrGroudTruth = self.VAEdecoder(symshapes, treekids, symparams, rd1)

        return myOut, paramOut, paramGT, NClrOut, NClrGroudTruth, mu, logvar