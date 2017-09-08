from __future__ import division, print_function
import torch
import model
from torch.autograd import Variable
import torch.nn as NN
import math

treekids = torch.ByteTensor([[[0,0,0],[0,0,0],[2,0,1],[1,3,0]]])
shapes = Variable(torch.DoubleTensor([[[1.0, 5.0], [2.0, 6.0]]]))
params = Variable(torch.DoubleTensor([[[0.0], [2.0], [0.0], [0.0]]]))

print(treekids.size())
print(shapes.size())
print(params.size())

class TestModel(torch.nn.Module):
    def __init__(self,  hiddenSize, latentSize, symSize, boxSize, catSize):
        super(TestModel,self).__init__()
        self.hidden=hiddenSize
        self.latent=latentSize
        self.sym=symSize
        self.box=boxSize
        self.cat = catSize
        self.hOut = list(Variable(torch.FloatTensor(1, self.latent).zero_(), requires_grad = True) for i in range(4))
        self.encoder = model.RvnnEncoCell(hiddenSize, latentSize, symSize, boxSize)
        self.decoder = model.RvnnDecoCell(hiddenSize, latentSize, symSize, boxSize, catSize)
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

    def Encode(self, kids, shape, param):
        nodeNum = kids.size(1)
        leafNum = shape.size(2)
        for i in range(nodeNum):
            if(kids[0,i,:][0] == 0):
                nodeType = 0
                inputa = shape[0, :, i].contiguous().view(1, -1).float()
                inputb = None
            elif(kids[0,i,:][0] != 0 and kids[0,i,:][2] == 0):
                nodeType = 1
                if(i > leafNum-1):
                    inputa = self.hOut[kids[0,i,0]-1]
                    inputb = self.hOut[kids[0,i,1]-1]
            elif(kids[0,i,:][0] != 0 and kids[0,i,:][2] == 1):
                nodeType = 2
                if(i > leafNum-1):
                    inputa = self.hOut[kids[0,i,0]-1]
                    inputb = param[:, kids[0,i,0]-1, :].float()

            self.hOut[i] = self.encoder(nodeType,inputa,inputb)

        return self.hOut[nodeNum-1]

    def concate(self, input1):
        Out = input1[0]
        for i in range(1,len(input1)):
            tmpOut = input1[i]
            Out = torch.cat((Out, tmpOut), 0)

        return Out

    def Decode(self, kids, shape, param, rd1):
        nodeNum = kids.size(1); leafNum = shape.size(2)
        paramOut = list(Variable(torch.FloatTensor(1, param.size(2)).zero_(), requires_grad = True) for i in range(param.size(1)))
        paramGT = list(Variable(torch.FloatTensor(1, param.size(2)).zero_(), requires_grad = True) for i in range(param.size(1)))
        NClrOut = list(Variable(torch.FloatTensor(1, self.cat).zero_(), requires_grad = True) for i in range(nodeNum))
        NClrGroundTruth = list(Variable(torch.LongTensor(1, 1).zero_(), requires_grad = True) for i in range(nodeNum))
        hmidOut = list(Variable(torch.FloatTensor(1, self.latent).zero_(), requires_grad = True) for i in range(nodeNum))
        hOut = list(Variable(torch.FloatTensor(1, self.box).zero_(), requires_grad = True) for i in range(leafNum))
        hmidOut[nodeNum - 1] = rd1

        for i in reversed(range(nodeNum)):
            nodeType = 0
            if (kids[0, i, :][0] == 0):
                nodeType = 0
                NClrGroundTruth[i] = Variable(torch.LongTensor([0]))
            elif(kids[0,i,:][0] != 0 and kids[0,i,:][2] == 0):
                #Adjacent Node
                nodeType = 1
                NClrGroundTruth[i] = Variable(torch.LongTensor([1]))
            elif(kids[0,i,:][0] != 0 and kids[0,i,:][2] == 1):
                #Symmetry Node
                nodeType = 2
                NClrGroundTruth[i] = Variable(torch.LongTensor([2]))
                paramGT[i] = param[0, kids[0, i, 0]-1, :].float()

            input1 = hmidOut[i]
            tmpOut = self.decoder(nodeType,input1)

            if(kids[0, i, :][0] == 0):
                hOut[i] = tmpOut.narrow(1, 0, self.box)
                NClrOut[i] = tmpOut.narrow(1, self.box, self.box+self.cat)
            elif(kids[0,i,:][0] != 0 and kids[0,i,:][2] == 0):
                id1 = kids[0, i, 0]
                id2 = kids[0, i, 1]
                hmidOut[id1-1] = tmpOut.narrow(1, 0, self.latent)
                hmidOut[id2-1] = tmpOut.narrow(1, self.latent, self.latent)
                NClrOut[i] = tmpOut.narrow(1, self.latent+self.latent, self.cat)
            elif(kids[0, i, :][0] != 0 and kids[0, i, :][2] == 1):
                id1 = kids[0, i, 0]
                hmidOut[id1-1] = tmpOut.narrow(1, 0, self.latent)
                paramOut[id1-1] = tmpOut.narrow(1, self.latent, self.sym)
                NClrOut[i] = tmpOut.narrow(1, self.latent + self.sym, self.cat)

        hOut = self.concate(hOut)
        paramOut = self.concate(paramOut)
        paramGT = self.concate(paramGT)
        NClrOut = self.concate(NClrOut)
        NClrGroundTruth = self.concate(NClrGroundTruth)
        return hOut, paramOut, paramGT, NClrOut, NClrGroundTruth

    def forward(self, kids, shape, param):
        encOutput = self.Encode(kids, shape, param)

        re1 = self.tanh(self.ranen1(encOutput))
        re2 = self.ranen2(re1)

        mu = re2.narrow(1, 0, self.latent)
        logvar = re2.narrow(1, self.latent, self.latent*2)

        sig = torch.exp(logvar/2)
        eps = Variable(torch.randn(mu.size()), requires_grad = True)
        sample_z = mu + sig * eps

        rd2 = self.tanh(self.rande2(sample_z))
        rd1 = self.tanh(self.rande1(rd2))

        hOut, paramOut, paramGT, NClrOut, NClrGroundTruth = self.Decode(kids,shape,param,rd1)

        return hOut, paramOut, paramGT, NClrOut, NClrGroundTruth

def TrainEnco():
    test = TestModel(6,4,1,2,3)
    test1 = model.RvnnEncoCell(6,4,1,2)
    torch.save(test.encoder.state_dict(),'test')
    test1.load_state_dict(torch.load('test'))
    Out = test(treekids, shapes, params)
    # tmp1 = test.hOut[0]
    # tmp = tmp1.data.numpy()
    # input1 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    # tmp2 = test.hOut[1]
    # tmp = tmp2.data.numpy()
    # input2 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    # tmp3 = test.hOut[2]
    # tmp = tmp3.data.numpy()
    # input3 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    # tmp4 = test.hOut[3]
    # tmp = tmp4.data.numpy()
    # input4 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    # Out.backward(torch.FloatTensor([[1,1,1,1]]))
    # Out1 = test1(1,input1,input3)
    # Out1.backward(torch.FloatTensor([[1,1,1,1]]))
    #
    # test3 = model.RvnnEncoCell(6,4,1,2)
    # test3.load_state_dict(torch.load('test'))
    # Out3 = test3(2,input2,params[:, treekids[0,2,0]-1, :].float())
    # input3grad = input3.grad
    # Out3.backward(input3grad)
    #
    # test2 = model.RvnnEncoCell(6,4,1,2)
    # test2.load_state_dict(torch.load('test'))
    # Out2 = test2(0,shapes[0, :, 0].contiguous().view(1, -1).float())
    # input1grad = input1.grad
    # Out2.backward(input1grad)
    # test4 = model.RvnnEncoCell(6,4,1,2)
    # test4.load_state_dict(torch.load('test'))
    # Out4 = test4(0,shapes[0, :, 1].contiguous().view(1, -1).float())
    # input2grad = input2.grad
    # Out4.backward(input2grad)

    print(Out)
    # print(tmp1)
    # print(Out2)
    # print(tmp2)
    # print(Out4)
    # print(tmp3)
    # print(Out3)
    # print(tmp4)
    # print(Out1)

def TrainDeco():
    test = TestModel(6,4,1,2,3)
    Out, paramOut, paramGT, NClrOut, NClrGT = test(treekids,shapes,params)
    tmpshape = test.concate(shapes.float())
    print(Out)
    print(paramOut)
    print(paramGT)
    print(NClrOut)
    print(NClrGT)

    # KLloss = torch.sum(1.0 + torch.log(torch.pow(sig, 2.0)) - torch.pow(mu, 2.0) - torch.pow(sig, 2.0)) * (-0.05)

    # calculate Node Classification Loss
    NClloss = torch.nn.functional.cross_entropy(NClrOut, NClrGT)

    # calculate Sym parameters Loss
    paramLoss = torch.nn.functional.mse_loss(paramOut, paramGT) * 0.2

    # calculate reconstruction Loss
    reconLoss = torch.nn.functional.mse_loss(Out, tmpshape) * 0.8

    # final Loss
    FinalLoss = NClloss + paramLoss + reconLoss

    FinalLoss.backward()

    print("hello")

# TrainEnco()
TrainDeco()