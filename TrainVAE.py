from __future__ import print_function, division
import math
import torch
import torch.nn as NN
import model
from torch.autograd import Variable
import torch.nn.functional as F
import Queue as queue
import numpy as np
from numpy import linalg as LA
from GBpy import tools

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


    def VAEdecoder(self, symshapes, treekids, symparams, rd1):
        nodeNum = treekids.size(1); leafNum = symshapes.size(2)
        paramOut = list(Variable(torch.FloatTensor(1, symparams.size(2)).zero_(), requires_grad = True) for i in range(symparams.size(1)))
        paramGT = list(Variable(torch.FloatTensor(1, symparams.size(2)).zero_(), requires_grad = True) for i in range(symparams.size(1)))
        NClrOut = list(Variable(torch.FloatTensor(1, self.cat).zero_(), requires_grad = True) for i in range(nodeNum))
        NClrGroundTruth = list(Variable(torch.LongTensor(1, 1).zero_(), requires_grad = True) for i in range(nodeNum))
        hmidOut = list(Variable(torch.FloatTensor(1, self.latent).zero_(), requires_grad = True) for i in range(nodeNum))
        hOut = list(Variable(torch.FloatTensor(1, self.box).zero_(), requires_grad = True) for i in range(leafNum))
        hmidOut[nodeNum - 1] = rd1

        for i in reversed(range(nodeNum)):
            nodeType = 0
            if (treekids[0, i, :][0] == 0):
                nodeType = 0
                NClrGroundTruth[i] = Variable(torch.LongTensor([0]))
            elif(treekids[0, i, :][0] != 0 and treekids[0, i,:][2] == 0):
                #Adjacent Node
                nodeType = 1
                NClrGroundTruth[i] = Variable(torch.LongTensor([1]))
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] == 1):
                #Symmetry Node
                nodeType = 2
                NClrGroundTruth[i] = Variable(torch.LongTensor([2]))
                paramGT[treekids[0, i, 0]-1] = symparams[:,treekids[0, i, 0]-1, :].float()

            input1 = hmidOut[i]
            tmpLeftOut, tmpRightOut, tmpClrOut = self.decoder(nodeType,input1)

            NClrOut[i] = tmpClrOut
            if(treekids[0, i, :][0] == 0):
                hOut[i] = tmpLeftOut
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] == 0):
                id1 = treekids[0, i, 0]
                id2 = treekids[0, i, 1]
                hmidOut[id1-1] = tmpLeftOut
                hmidOut[id2-1] = tmpRightOut
            elif(treekids[0, i, :][0] != 0 and treekids[0, i, :][2] != 0):
                id1 = treekids[0, i, 0]
                hmidOut[id1-1] = tmpLeftOut
                paramOut[id1-1] = tmpRightOut

        hOut = torch.stack(hOut).squeeze()
        paramOut = torch.stack(paramOut).squeeze()
        paramGT = torch.stack(paramGT).squeeze()
        NClrOut = torch.stack(NClrOut).squeeze()
        NClrGroundTruth = torch.stack(NClrGroundTruth).squeeze()
        return hOut, paramOut, paramGT, NClrOut, NClrGroundTruth

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def testVAE(self, num, symshapes, treekids, symparams):
        genShapes = list([] for jj in range(num))
        for i in range(num):
            encOutput = self.VAEencoder(symshapes, treekids, symparams)
            re1 = self.tanh(self.ranen1(encOutput))
            re2 = self.ranen2(re1)

            mu = re2.narrow(1, 0, self.latent)
            logvar = re2.narrow(1, self.latent, self.latent)

            sample_z = self.reparameterize(mu, logvar)

            # sample_z = torch.FloatTensor(1, self.latent).normal_()
            # sample_z = Variable(sample_z)

            rd2 = self.tanh(self.rande2(sample_z))
            rd1 = self.tanh(self.rande1(rd2))

            feature = queue.Queue(maxsize = 40)
            symlist = queue.Queue(maxsize = 40)
            feature.put(rd1)
            symlist.put(np.ones((1,10))*10)
            while not feature.empty():
                p = feature.get()
                sm = self.decoder.getClass(p)
                l_index = np.argmax(sm.data.numpy())
                tmpLeftOut, tmpRightOut, NClr = self.decoder(l_index,p)
                if(l_index == 0):
                    re_box = tmpLeftOut
                    re_box = re_box.data.numpy()
                    symfeature = symlist.get()
                    genShapes[i].append(re_box.squeeze())
                    if(abs(symfeature[:,0] + 1.0) < 0.15):
                        folds = 1//symfeature[7]
                        new_box = re_box.copy()
                        symfeature[1:4] = symfeature[1:4]/LA.norm(symfeature[1:4])
                        for kk in range(1, folds):
                            rotvector = np.append(symfeature[1:4],[symfeature[7]*2*math.pi*kk, 1])
                            rotm = tools.vrrotmat2vec(rotvector)
                            center = re_box[:,0:3]
                            dir_1 = re_box[:,6:9]
                            dir_2 = re_box[:,9:12]
                            newcenter = np.dot(rotm, center-symfeature[4:7]) + symfeature[4:7]
                            new_box[:,0:3] = newcenter
                            new_box[:,6:9] = np.dot(rotm, dir_1)
                            new_box[:,9:12] = np.dot(rotm, dir_2)
                            genShapes[i].append(new_box.squeeze())
                    if(abs(symfeature[:,0]) < 0.15):
                        trans = symfeature[:,1:4]
                        trans_end = symfeature[:,4:7]
                        center = re_box[:,0:3]
                        trans_length = LA.norm(trans)
                        trans_total = LA.norm(trans_end-center)
                        folds = trans_total/trans_length
                        for kk in range(folds):
                            new_box = re_box.copy()
                            newcenter = center + kk * trans
                            new_box[:,0:3] = newcenter
                            genShapes[i].append(new_box.squeeze())
                    if(abs(symfeature[:,0] -1) < 0.15):
                        ref_normal = symfeature[:,1:4]
                        ref_normal = ref_normal/LA.norm(ref_normal)
                        ref_point = symfeature[:,4:7]
                        new_box = re_box.copy()
                        center = re_box[:, 0:3]
                        if(np.dot(ref_normal, np.transpose(ref_point-center)) < 0):
                            ref_normal = -1*ref_normal

                        newcenter = abs(sum((ref_point-center) * ref_normal))*ref_normal*2 + center
                        new_box[:, 0:3] = newcenter

                        dir_1 = re_box[:,6:9]
                        if(np.dot(ref_normal,np.transpose(dir_1)) > 0):
                            ref_normal = -1 * ref_normal

                        new_box[:, 6:9] = dir_1 - 2*np.dot(dir_1, np.transpose(ref_normal))* ref_normal

                        dir_2 = re_box[:, 9:12]
                        if(np.dot(ref_normal, np.transpose(dir_2)) > 0):
                            ref_normal = -ref_normal

                        new_box[:, 9:12] = dir_2 - 2*np.dot(dir_2, np.transpose(ref_normal))*ref_normal

                        genShapes[i].append(new_box.squeeze())
                else:
                    if(l_index == 2):
                        y1 = tmpLeftOut
                        feature.put(y1)
                        symfeature = tmpRightOut
                        symlist.get()
                        symlist.put(symfeature.data.numpy())
                    elif(l_index == 1):
                        y1 = tmpLeftOut
                        y2 = tmpRightOut
                        feature.put(y1)
                        feature.put(y2)
                        tmp1 = symlist.get()
                        symlist.put(tmp1)
                        symlist.put(tmp1)

        return genShapes


    def forward(self, symshapes, treekids, symparams):
        encOutput = self.VAEencoder(symshapes, treekids, symparams)
        re1 = self.tanh(self.ranen1(encOutput))
        re2 = self.ranen2(re1)

        mu = re2.narrow(1, 0, self.latent)
        logvar = re2.narrow(1, self.latent, self.latent)

        sample_z = self.reparameterize(mu, logvar)

        rd2 = self.tanh(self.rande2(sample_z))
        rd1 = self.tanh(self.rande1(rd2))

        myOut, paramOut, paramGT, NClrOut, NClrGroudTruth = self.VAEdecoder(symshapes, treekids, symparams, rd1)

        return myOut, paramOut, paramGT, NClrOut, NClrGroudTruth, mu, logvar