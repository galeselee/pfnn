# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Train NetG and NetF/NetLoss"""
from mindspore.train.model import Model
from mindspore import save_checkpoint, load_param_into_net, load_checkpoint
from mindspore.train.callback import Callback
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np


class SaveCallbackNETG(Callback):
    """
    SavedCall for NetG to print loss and save checkpoint

    Args:
        net(NetG): The instantiation of NetG
        path(str): The path to save the checkpoint of NetG
    """
    def __init__(self, net, path):
        super(SaveCallbackNETG, self).__init__()
        self.loss = 1e5
        self.net = net
        self.path = path
        self.print = ops.Print()

    def step_end(self, run_context):
        """print info and save checkpoint per 100 steps"""
        cb_params = run_context.original_args()
        if bool(cb_params.net_outputs < self.loss) and cb_params.cur_epoch_num % 100 == 0:
            self.loss = cb_params.net_outputs
            save_checkpoint(self.net, self.path)
        if cb_params.cur_epoch_num % 100 == 0:
            self.print(
                f"NETG epoch : {cb_params.cur_epoch_num}, loss : {cb_params.net_outputs}")


class SaveCallbackNETLoss(Callback):
    """
    SavedCall for NetG to print loss and save checkpoint

    Args:
        net(NetG): The instantiation of NetF
        path(str): The path to save the checkpoint of NetF
        x(np.array): valid dataset
        ua(np.array): Label of valid dataset
    """
    def __init__(self, net, path, x, l, g, ua):
        super(SaveCallbackNETLoss, self).__init__()
        self.loss = 1e5
        self.error = 1e5
        self.net = net
        self.path = path
        self.l = l
        self.x = x
        self.g = g
        self.ua = ua
        self.print = ops.Print()

    def step_end(self, run_context):
        """print info and save checkpoint per 100 steps"""
        cb_params = run_context.original_args()
        u = (Tensor(self.g, mstype.float32) + Tensor(self.l, mstype.float32)
             * self.net(Tensor(self.x, mstype.float32))).asnumpy()
        self.tmp_error = (((u - self.ua)**2).sum()/self.ua.sum())**0.5
        if self.error > self.tmp_error and cb_params.cur_epoch_num % 100 == 0:
            self.error = self.tmp_error
            save_checkpoint(self.net, self.path)
        self.loss = cb_params.net_outputs
        if cb_params.cur_epoch_num % 100 == 0:
            self.print(
                f"NETF epoch : {cb_params.cur_epoch_num}, loss : {self.loss}, error : {self.tmp_error}")


def train_netg(args, net, optim, dataset):
    """
    The process of preprocess and process to train NetG

    Args:
        net(NetG): The instantiation of NetG
        optim: The optimizer to optimal NetG
        dataset: The traing dataset of NetG
    """
    print("START TRAIN NEURAL NETWORK G")
    model = Model(network=net, loss_fn=None, optimizer=optim)
    model.train(args.g_epochs, dataset, callbacks=[
                SaveCallbackNETG(net, args.g_path)])


def train_netloss(args, netg, netf, netloss, lenfac, optim,
                  InSet, BdSet, dataset):
    """
    The process of preprocess and process to train NetF/NetLoss

    Args:
        netg: The Instantiation of NetG
        netf: The Instantiation of NetF
        netloss: The Instantiation of NetLoss
        lenfac: The Instantiation of lenfac
        optim: The optimizer to optimal NetF/NetLoss
        dataset: The trainging dataset of NetF/NetLoss
    """
    grad_ = ops.composite.GradOperation(get_all=True)

    InSet_l = lenfac(Tensor(InSet.x[:, 0], mstype.float32))
    InSet_l = InSet_l.reshape((len(InSet_l), 1))
    InSet_lx = grad_(lenfac)(Tensor(InSet.x[:, 0], mstype.float32))[
        0].asnumpy()[:, np.newaxis]
    InSet_lx = np.hstack((InSet_lx, np.zeros(InSet_lx.shape)))
    BdSet_nl = lenfac(Tensor(BdSet.n_x[:, 0], mstype.float32)).asnumpy()[
        :, np.newaxis]

    load_param_into_net(netg, load_checkpoint(args.g_path), strict_load=True)
    InSet_g = netg(Tensor(InSet.x, mstype.float32))
    InSet_gx = grad_(netg)(Tensor(InSet.x, mstype.float32))[0]
    BdSet_ng = netg(Tensor(BdSet.n_x, mstype.float32))

    netloss.get_variable(InSet_g, InSet_l, InSet_gx, InSet_lx, InSet.a, InSet.size,
                         InSet.dim, InSet.area, InSet.c, BdSet.n_length, BdSet.n_r, BdSet_nl, BdSet_ng)
    print("START TRAIN NEURAL NETWORK F")
    model = Model(network=netloss, loss_fn=None, optimizer=optim)
    model.train(args.f_epochs, dataset, callbacks=[SaveCallbackNETLoss(
        netf, args.f_path, InSet.x, InSet_l, InSet_g, InSet.ua)])


def train(args, netg, netf, netloss, lenfac, optim_g, optim_f,
          InSet, BdSet, dataset_g, dataset_loss):
    """
    The traing process that's includes traning network G and network F/Loss
    """
    print("START TRAINING")
    train_netg(args, netg, optim_g, dataset_g)
    train_netloss(args, netg, netf, netloss, lenfac, optim_f,
                  InSet, BdSet, dataset_loss)
