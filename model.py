from __future__ import print_function, division
import torch
import math
import torch.nn as NN

class RvnnEncoCell(NN.Module):
    def __init__(self,hiddenSize,latentSize,symSize,boxSize):
        '''
            basic Module for calculate
        '''
        super(RvnnEncoCell, self).__init__()
        self.hidden=hiddenSize
        self.latent=latentSize
        self.sym=symSize
        self.box=boxSize
        #Tanh operation
        self.tanh = NN.Tanh()
        #sym tree node
        self.symEnco1 = NN.Linear(self.sym+self.latent,self.hidden)
        self.symEnco2 = NN.Linear(self.hidden,self.latent)
        #Adjacent node
        self.AdjEnco1 = NN.Linear(self.latent+self.latent,self.hidden)
        self.AdjEnco2 = NN.Linear(self.hidden,self.latent)
        #Box Encoder node
        self.BoxEnco = NN.Linear(self.box,self.latent)

        #initialization of parameters in Modules
        for i,m in enumerate(self.modules()):
            if(i != 0):
                if(not isinstance(m, NN.Tanh)):
                    r = math.sqrt(6/(self.hidden+self.hidden+1))
                    m.weight.data = torch.rand(m.weight.data.size())*2*r-r

    def forward(self,treeNodeType,input1,input2=None):
        #forward operation in Encode Tree of VAE
        # LeafNode below:
        if(treeNodeType == 0):
            myOut = self.tanh(self.BoxEnco(input1))
        # Adjacent Node:
        elif(treeNodeType == 1):
            tmpInput = torch.cat((input1,input2), 1)
            hiddenOut = self.tanh(self.AdjEnco1(tmpInput))
            myOut = self.tanh(self.AdjEnco2(hiddenOut))
        #Symmetry Node:
        elif(treeNodeType == 2):
            tmpInput = torch.cat((input1,input2), 1)
            hiddenOut = self.tanh(self.symEnco1(tmpInput))
            myOut = self.tanh(self.symEnco2(hiddenOut))

        return myOut

class RvnnDecoCell(NN.Module):
    '''
        Rvnn Decoder Cell in Decoder of VAE
    '''
    def __init__(self, hiddenSize, latentSize, symSize, boxSize, catSize):
        super(RvnnDecoCell, self).__init__()
        self.gAssemcount=0.0; self.gSymcount=0.0; self.gLeafcount=0.0
        self.hidden = hiddenSize
        self.latent = latentSize
        self.sym = symSize
        self.box = boxSize
        self.cat = catSize
        #Tanh operation
        self.tanh = NN.Tanh()
        #sym tree node
        self.symDeco1 = NN.Linear(self.hidden, self.sym+self.latent)
        self.symDeco2 = NN.Linear(self.latent, self.hidden)
        #Adjacent node
        self.AdjDeco1 = NN.Linear(self.hidden, self.latent+self.latent)
        self.AdjDeco2 = NN.Linear(self.latent, self.hidden)
        #Box Encoder node
        self.BoxDeco = NN.Linear(self.latent, self.box)
        #Node Classifer
        self.NClr1 = NN.Linear(self.latent, self.hidden)
        self.NClr2 = NN.Linear(self.hidden, self.cat)

        #initialization of parameters in Modules
        for i, m in enumerate(self.modules()):
            if(i != 0):
                if(not isinstance(m, NN.Tanh)):
                    r = math.sqrt(6/(self.hidden+self.hidden+1))
                    m.weight.data = torch.rand(m.weight.data.size())*2*r-r

    def getClass(self, latentCode):
        ClrHiddenOut = self.tanh(self.NClr1(latentCode))
        ClrOut = self.NClr2(ClrHiddenOut)
        myOut = NN.functional.softmax(ClrOut)

        return myOut

    def forward(self,treeNodeType,input1):
        ClrHiddenOut = self.tanh(self.NClr1(input1))
        ClrOut = self.NClr2(ClrHiddenOut)
        LeftOut =None
        RightOut=None
        if(treeNodeType == 0):
            myOut=self.tanh(self.BoxDeco(input1))
            LeftOut = myOut
            self.gLeafcount = self.gLeafcount + 1.0
        elif(treeNodeType == 1):
            hiddenOut = self.tanh(self.AdjDeco2(input1))
            myOut = self.tanh(self.AdjDeco1(hiddenOut))
            LeftOut = myOut.narrow(1, 0, self.latent)
            RightOut = myOut.narrow(1, self.latent, self.latent)
            self.gAssemcount = self.gAssemcount + 1.0
        elif(treeNodeType == 2):
            hiddenOut = self.tanh(self.symDeco2(input1))
            myOut = self.tanh(self.symDeco1(hiddenOut))
            LeftOut = myOut.narrow(1, 0, self.latent)
            RightOut = myOut.narrow(1, self.latent, self.sym)
            self.gSymcount = self.gSymcount + 1.0


        return LeftOut, RightOut, ClrOut