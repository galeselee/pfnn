
from mindspore.train.model import Model
from mindspore import save_checkpoint, load_param_into_net, load_checkpoint
from mindspore.train.callback import Callback
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np

class SaveCallback(Callback):
    def __init__(self, net, path):
        super(SaveCallback,self).__init__()
        self.loss = 1e5
        self.net = net
        self.path = path
    
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if bool(cb_params.net_outputs < self.loss) and cb_params.cur_epoch_num % 100 == 0 :
            self.loss = cb_params.net_outputs
            save_checkpoint(self.net, self.path)


def train_netg(args, net, optim, dataset):
    print("START TRAIN NEURAL NETWORK G")
    model = Model(network=net, loss_fn=None, optimizer=optim)
    model.train(args.g_epochs, dataset, callbacks=[SaveCallback(net, args.g_path)])


def train_netloss(args, netg, netf, netloss, lenfac, optim,\
                 InSet, BdSet, dataset):
    grad_ = ops.composite.GradOperation(get_all=True)
    
    InSet_l = lenfac(Tensor(InSet.x[:,0],mstype.float32))
    InSet_l = InSet_l.reshape((len(InSet_l),1))
    InSet_lx = grad_(lenfac)(Tensor(InSet.x[:,0],mstype.float32))[0].asnumpy()[:,np.newaxis]
    InSet_lx = np.hstack((InSet_lx, np.zeros(InSet_lx.shape)))
    BdSet_nl = lenfac(Tensor(BdSet.n_x[:,0],mstype.float32)).asnumpy()[:,np.newaxis]

    load_param_into_net(netg,load_checkpoint(args.g_path) ,strict_load=True)
    InSet_g = netg(Tensor(InSet.x,mstype.float32))
    InSet_gx = grad_(netg)(Tensor(InSet.x,mstype.float32))[0]
    BdSet_ng = netg(Tensor(BdSet.n_x,mstype.float32))

    netloss.get_variable(InSet_g, InSet_l, InSet_gx, InSet_lx, InSet.a, InSet.size, \
       InSet.dim, InSet.area, InSet.c, BdSet.n_length, BdSet.n_r, BdSet_nl, BdSet_ng)
    print("START TRAIN NEURAL NETWORK F")
    model=Model(network=netloss,loss_fn=None,optimizer=optim)
    model.train(args.f_epochs, dataset, callbacks=[SaveCallback(netf, args.f_path)])


def train(args, netg, netf, netloss, lenfac, optim_g, optim_f, \
          InSet, BdSet, dataset_g, dataset_loss):
    print("STRAT TRAINING")
    train_netg(args, netg, optim_g, dataset_g)
    train_netloss(args, netg, netf, netloss, lenfac, optim_f, \
                InSet, BdSet, dataset_loss)
