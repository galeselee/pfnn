import mindspore.dataset as ds
import numpy as np

class DataSetNetG:
    def __init__(self, data, label):
        self.data = data
        self.label = label 
    def __getitem__(self, index):
        return (self.data, self.label)
    def __len__(self):
        return 1


class DataSetNetLoss:
    def __init__(self, x, nx):
        self.x = x
        self.nx = nx
    
    def __getitem__(self, index):
        return (self.x, self.nx)

    def __len__(self):
        return 1
    
def GenerateDataSet(inset, bdset):
    datasetnetg = DataSetNetG(bdset.d_x, bdset.d_r)
    datasetnetloss = DataSetNetLoss(inset.x, bdset.n_x) 
    DS_NETG = ds.GeneratorDataset(datasetnetg, ["data","label"], shuffle=False)
    DS_NETL = ds.GeneratorDataset(datasetnetloss, ["x_inset","x_bdset"], shuffle=False)
    return DS_NETG, DS_NETL
