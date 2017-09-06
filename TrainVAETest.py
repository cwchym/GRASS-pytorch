from __future__ import division, print_function
import torch
import model
from torch.autograd import Variable

treekids = torch.ByteTensor([[[0,0,0],[0,0,0],[2,0,1],[1,3,0]]])
shapes = Variable(torch.DoubleTensor([[[1.0, 5.0], [2.0, 6.0]]]))
params = Variable(torch.DoubleTensor([[[0.0], [2.0], [0.0], [0.0]]]))

print(treekids.size())
print(shapes.size())
print(params.size())

class TestModel(torch.nn.Module):
    def __init__(self,  hiddenSize, latentSize, symSize, boxSize):
        super(TestModel,self).__init__()
        self.hidden=hiddenSize
        self.latent=latentSize
        self.sym=symSize
        self.box=boxSize
        self.hOut = list(Variable(torch.FloatTensor(1, self.latent).zero_()) for i in range(4))
        self.encoder = model.RvnnEncoCell(hiddenSize, latentSize, symSize, boxSize)

    def forward(self, kids, shape, param):
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

def Train():
    test = TestModel(6,4,1,2)
    test1 = model.RvnnEncoCell(6,4,1,2)
    torch.save(test.encoder.state_dict(),'test')
    test1.load_state_dict(torch.load('test'))
    Out = test(treekids,shapes,params)
    tmp1 = test.hOut[0]
    tmp = tmp1.data.numpy()
    input1 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    tmp2 = test.hOut[1]
    tmp = tmp2.data.numpy()
    input2 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    tmp3 = test.hOut[2]
    tmp = tmp3.data.numpy()
    input3 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    tmp4 = test.hOut[3]
    tmp = tmp4.data.numpy()
    input4 = Variable(torch.FloatTensor(tmp),requires_grad = True)
    Out.backward(torch.FloatTensor([[1,1,1,1]]))
    Out1 = test1(1,input1,input3)
    Out1.backward(torch.FloatTensor([[1,1,1,1]]))

    test3 = model.RvnnEncoCell(6,4,1,2)
    test3.load_state_dict(torch.load('test'))
    Out3 = test3(2,input2,params[:, treekids[0,2,0]-1, :].float())
    input3grad = input3.grad
    Out3.backward(input3grad)

    test2 = model.RvnnEncoCell(6,4,1,2)
    test2.load_state_dict(torch.load('test'))
    Out2 = test2(0,shapes[0, :, 0].contiguous().view(1, -1).float())
    input1grad = input1.grad
    Out2.backward(input1grad)
    test4 = model.RvnnEncoCell(6,4,1,2)
    test4.load_state_dict(torch.load('test'))
    Out4 = test4(0,shapes[0, :, 1].contiguous().view(1, -1).float())
    input2grad = input2.grad
    Out4.backward(input2grad)

    print(Out)
    print(tmp1)
    print(Out2)
    print(tmp2)
    print(Out4)
    print(tmp3)
    print(Out3)
    print(tmp4)
    print(Out1)



Train()