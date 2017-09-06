import torch
import torch.nn as NN
from TrainVAE import myVAE
from data_loader import get_loader
from torch.autograd import Variable

def Train(path, batch, shuffle, hiddenSize, latentSize, symSize, boxSize, catSize):
    data = get_loader(path, batch, shuffle)

    for i in range(1500):
        for j,d in enumerate(data):
            symshapes = Variable(d['symshapes'])
            treekids = d['treekids']
            symparams = Variable(d['symparams'])
            VAE = myVAE(symshapes, treekids, symparams, hiddenSize, latentSize, symSize, boxSize, catSize)

            #calculate Output!!!
            Out, paramOut, paramGT, NClrOut, NClrGT, mu, sig = VAE()

            #calculate KL Loss
            KLloss = torch.sum(1.0+torch.log(torch.pow(sig,2.0))-torch.pow(mu,2.0)-torch.pow(sig,2.0))*(-0.05)

            #calculate Node Classification Loss
            NClloss = torch.sum(torch.nn.functional.cross_entropy(NClrOut,NClrGT))

            #calculate Sym parameters Loss
            paramLoss = torch.sum(torch.nn.functional.mse_loss(paramOut, paramGT))*0.2

            #calculate reconstruction Loss
            reconLoss = torch.sum(torch.nn.functional.mse_loss(Out, symshapes))*0.8

            #final Loss
            FinalLoss = KLloss + NClloss + paramLoss + reconLoss
            leafcount = VAE.decoder.gLeafcount
            gAssemcount = VAE.decoder.gAssemcount
            gSymcount = VAE.decoder.gSymcount
            # optimization = torch.optim.SGD([{'params':VAE.encoder.parameters(), 'lr':0.2}, {'params':VAE.decoder.parameters(),'lr':0.2}, ], lr=0.2)
            optimization = torch.optim.SGD(
                [{'params': VAE.encoder.BoxEnco.parameters(), 'lr': 0.2/leafcount},
                 {'params': VAE.encoder.symEnco1.parameters(), 'lr':0.2/gSymcount},
                 {'params': VAE.encoder.symEnco2.parameters(), 'lr':0.2/gSymcount},
                 {'params': VAE.encoder.AdjEnco1.parameters(), 'lr':0.2/gAssemcount},
                 {'params': VAE.encoder.AdjEnco2.parameters(), 'lr':0.2/gAssemcount},
                 {'params': VAE.decoder.BoxDeco.parameters(), 'lr': 0.2/leafcount},
                 {'params': VAE.decoder.symDeco1.parameters(), 'lr':0.2/gSymcount},
                 {'params': VAE.decoder.symDeco2.parameters(), 'lr':0.2/gSymcount},
                 {'params': VAE.decoder.AdjDeco1.parameters(), 'lr':0.2/gAssemcount},
                 {'params': VAE.decoder.AdjDeco2.parameters(), 'lr':0.2/gAssemcount},
                 {'params': VAE.decoder.NClr1.parameters(), 'lr':0.2/treekids.size(0)},
                 {'params': VAE.decoder.NClr2.parameters(), 'lr':0.2/treekids.size(0)},
                 {'params': VAE.ranen1.parameters()},
                 {'params': VAE.ranen2.parameters()},
                 {'params': VAE.rande1.parameters()},
                 {'params': VAE.rande2.parameters()}],
                lr=0.2)

            optimization.zero_grad()
            FinalLoss.backward()

            optimization.step()