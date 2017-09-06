import model
import torch
from torch.autograd import Variable

m = model.RvnnEncoCell(6,4,1,2)
inputa1 = Variable(torch.FloatTensor([[1.0,2.0,3.0,4.0]]))
inputa2 = Variable(torch.FloatTensor([[5.0,6.0,7.0,8.0]]))
print(torch.cat((inputa1,inputa2),1))
Out = m(0,inputa1, inputa2)
print(Out)