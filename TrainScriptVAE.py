import torch
import torch.nn as NN
from TrainVAE import myVAE
from data_loader import get_loader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib

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
    optimization = torch.optim.SGD(
        [{'params': VAE.encoder.BoxEnco.parameters()},
         {'params': VAE.decoder.BoxDeco.parameters()},
         {'params': VAE.encoder.symEnco1.parameters()},
         {'params': VAE.encoder.symEnco2.parameters()},
         {'params': VAE.decoder.symDeco1.parameters()},
         {'params': VAE.decoder.symDeco2.parameters()},
         {'params': VAE.encoder.AdjEnco1.parameters()},
         {'params': VAE.encoder.AdjEnco2.parameters()},
         {'params': VAE.decoder.AdjDeco1.parameters()},
         {'params': VAE.decoder.AdjDeco2.parameters()},
         {'params': VAE.decoder.NClr1.parameters()},
         {'params': VAE.decoder.NClr2.parameters()},
         {'params': VAE.ranen1.parameters()},
         {'params': VAE.ranen2.parameters()},
         {'params': VAE.rande1.parameters()},
         {'params': VAE.rande2.parameters()}],
        lr=0.2/20)

    histLoss = list()

    for i in range(1500):
        batchLoss = list()
        for j,d in enumerate(data):
            symshapes = Variable(d['symshapes'])
            treekids = d['treekids']
            symparams = Variable(d['symparams'])

            #calculate Output!!!
            myOut, paramOut, paramGT, NClrOut, NClrGT, mu, sig = VAE(symshapes, treekids, symparams)

            #calculate KL Loss
            KLloss = torch.sum(1.0+torch.log(torch.pow(sig,2.0))-torch.pow(mu,2.0)-torch.pow(sig,2.0))*(-0.05)

            #calculate Node Classification Loss
            NClloss = torch.nn.functional.cross_entropy(NClrOut,NClrGT)*0.2

            #calculate Sym parameters Loss
            paramLoss = torch.nn.functional.mse_loss(paramOut, paramGT)

            #calculate reconstruction Loss
            tmpshapes = VAE.concate(symshapes.float())
            reconLoss = torch.nn.functional.mse_loss(myOut, tmpshapes)*0.8

            #final Loss
            FinalLoss = KLloss + NClloss + paramLoss + reconLoss
            batchLoss.append(FinalLoss.data.numpy())
            leafcount = VAE.decoder.gLeafcount
            gAssemcount = VAE.decoder.gAssemcount
            gSymcount = VAE.decoder.gSymcount
            # optimization = torch.optim.SGD([{'params':VAE.encoder.parameters(), 'lr':0.2}, {'params':VAE.decoder.parameters(),'lr':0.2}, ], lr=0.2)

            for k in range(len(optimization.param_groups)):
                if(k <= 1):
                    optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/leafcount
                elif(k <= 5):
                    optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/gSymcount
                elif(k <= 9):
                    optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/gAssemcount
                elif(k <= 11):
                    optimization.param_groups[k]['lr'] = optimization.param_groups[k]['lr']/treekids.size(1)

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

    torch.save(VAE.state_dict(),'VAE.pkl')

Train('data/trainingData_chair.mat', 1, True, 200, 80, 8, 12, 3)