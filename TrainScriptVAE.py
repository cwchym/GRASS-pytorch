import torch
import torch.nn as NN
from TrainVAE import myVAE
from data_loader import get_loader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def update_line(hl, ax, new_data):
    hl.set_xdata(range(len(new_data)))
    hl.set_ydata(new_data)
    plt.draw()
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.1)

def Train(path, batch, shuffle, hiddenSize, latentSize, symSize, boxSize, catSize):
    matplotlib.use('Qt5Agg')
    plt.ion()
    ax = plt.gca()
    ax.set_autoscale_on(True)
    hl,  = plt.plot([], [])

    data = get_loader(path, batch, shuffle)

    VAE = myVAE(hiddenSize, latentSize, symSize, boxSize, catSize)
    # VAE.load_state_dict(torch.load('VAE.pkl'))
    # optimization = torch.optim.SGD(
    #     [{'params': VAE.encoder.BoxEnco.parameters()},
    #      {'params': VAE.decoder.BoxDeco.parameters()},
    #      {'params': VAE.encoder.symEnco1.parameters()},
    #      {'params': VAE.encoder.symEnco2.parameters()},
    #      {'params': VAE.decoder.symDeco1.parameters()},
    #      {'params': VAE.decoder.symDeco2.parameters()},
    #      {'params': VAE.encoder.AdjEnco1.parameters()},
    #      {'params': VAE.encoder.AdjEnco2.parameters()},
    #      {'params': VAE.decoder.AdjDeco1.parameters()},
    #      {'params': VAE.decoder.AdjDeco2.parameters()},
    #      {'params': VAE.decoder.NClr1.parameters(), 'lr':0.5/20},
    #      {'params': VAE.decoder.NClr2.parameters(), 'lr':0.5/20},
    #      {'params': VAE.ranen1.parameters()},
    #      {'params': VAE.ranen2.parameters()},
    #      {'params': VAE.rande1.parameters()},
    #      {'params': VAE.rande2.parameters()}],
    #      lr=0.2/20)
    optimization = torch.optim.SGD(VAE.parameters(), lr = 0.2/20)

    histLoss = list()

    for i in range(1500):
        batchLoss = list()
        for j,d in enumerate(data):
            boxes = d['boxes']
            symshapes = Variable(d['symshapes'].float())
            treekids = d['treekids']
            symparams = Variable(d['symparams'])

            #calculate Output!!!
            myOut, paramOut, paramGT, NClrOut, NClrGT, mu, logvar = VAE(symshapes, treekids, symparams)

            #calculate KL Loss
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLloss = torch.sum(KLD_element).mul(-0.5)

            # calculate Node Classification Loss
            finalNCl = torch.nn.functional.cross_entropy(NClrOut, NClrGT).mul_(0.2).mul_(NClrGT.size(0))

            # finalNCl = Variable(torch.FloatTensor([0]))
            # for ii in range(np.shape(NClrGT)[0]):
            #     finalNCl.add_(torch.nn.functional.cross_entropy(NClrOut[ii], NClrGT[ii]).mul_(0.2))

            # #calculate Sym parameters Loss
            finalPL = torch.nn.functional.mse_loss(paramOut, paramGT).mul_(paramGT.size(0))

            # finalPL = Variable(torch.FloatTensor([0]))
            # for ii in range(np.shape(paramGT)[0]):
            #     finalPL.add_(torch.nn.functional.mse_loss(paramOut[ii], paramGT[ii]))

            #calculate reconstruction Loss
            symshapes = torch.t(symshapes.squeeze())
            finalRL = torch.nn.functional.mse_loss(myOut, symshapes).mul_(0.8).mul_(myOut.size(0))

            # finalRL = Variable(torch.FloatTensor([0]))
            # for ii in range(np.shape(myOut)[0]):
            #     finalRL.add_(torch.nn.functional.mse_loss(myOut[ii], symshapes[:, :, ii]).mul_(0.8))

            #final Loss
            FinalLoss = Variable(torch.FloatTensor([0]))
            FinalLoss = FinalLoss.add_(KLloss).add_(finalNCl).add_(finalPL).add_(finalRL)
            batchLoss.append(FinalLoss.data.numpy())
            # leafcount = VAE.decoder.gLeafcount
            # gAssemcount = VAE.decoder.gAssemcount
            # gSymcount = VAE.decoder.gSymcount
            # optimization = torch.optim.SGD([{'params':VAE.encoder.parameters(), 'lr':0.2}, {'params':VAE.decoder.parameters(),'lr':0.2}, ], lr=0.2)

            # for k in range(len(optimization.param_groups)):
            #     if(k <= 1):
            #         optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/leafcount
            #     elif(k <= 5):
            #         optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/gSymcount
            #     elif(k <= 9):
            #         optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/gAssemcount
            #     elif(k <= 11):
            #         optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/treekids.size(1)

            if(j % 20 == 0 and j != 0):
                optimization.step()
                optimization.zero_grad()

                tmp = sum(batchLoss)/20
                histLoss.append(tmp)
                print('Epoch [%d/%d],  Iter [%d/%d], Loss: %.4f, ' % (i, 1500, j, len(data), tmp))
                del batchLoss[:]
                batchLoss = list()

            FinalLoss.backward()


        update_line(hl, ax, histLoss)

        if(i % 10 == 0 and i != 0):
            torch.save(VAE.state_dict(),'VAE.pkl')
            # for k in range(len(optimization.param_groups)):
            #     optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/2

Train('data/trainingData_chair.mat', 1, True, 200, 80, 8, 12, 3)