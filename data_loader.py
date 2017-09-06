from __future__ import print_function
import torch
import torch.utils.data as data
import scipy.io as sio
from torchvision import transforms
import numpy as np

class TrainMatData(data.Dataset):
    """
        Read mat data from trainingData_chair to program and compatible with torch.utils.data.DataLoader iterator
        Only ToTensor Transform
    """
    def __init__(self,matPath,transform=None):
        super(TrainMatData,self).__init__()
        matData = sio.loadmat(matPath, struct_as_record=False)
        self.data = matData['data']
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[0,item]
        boxes = data[0,0].boxes
        symshapes = data[0,0].symshapes
        treekids = data[0,0].treekids
        symparams = data[0,0].symparams

        tmp_null = np.array([[0,0,0,0,0,0,0,0]])
        tmp_out = None
        for i in symparams[0]:
            if(i.shape != (0,0)):
                if(tmp_out is None):
                    tmp_out = i
                else:
                    tmp_out=np.append(tmp_out,i,0)
            else:
                if(tmp_out is None):
                    tmp_out = tmp_null
                else:
                    tmp_out = np.append(tmp_out,tmp_null,0)

        sample = {'boxes':boxes, 'symshapes':symshapes, 'treekids':treekids, 'symparams':tmp_out}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.data.shape[1]

class ToTensor(object):
    """Convert Numpy data to Tensor"""

    def __call__(self, sample):
        boxes, symshapes, treekids, symparams = sample['boxes'], sample['symshapes'], sample['treekids'], sample['symparams']

        return {'boxes':torch.from_numpy(boxes),
                'symshapes':torch.from_numpy(symshapes),
                'treekids':torch.from_numpy(treekids),
                'symparams':torch.from_numpy(symparams)}


def get_loader(filePath,batchsize,shuffle,transform=transforms.Compose([ToTensor()])):
    trainingdata_chair = TrainMatData(matPath=filePath,
                                      transform=transform)

    m_data_loader = data.DataLoader(dataset=trainingdata_chair,
                                    batch_size=batchsize,
                                    shuffle=shuffle)

    return m_data_loader
