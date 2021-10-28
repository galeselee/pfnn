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
"""Run PFNN"""
import argparse
import time
import numpy as np
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import load_param_into_net, load_checkpoint
from src import DataSet, model, evalerror, train
from Data import Data


def ArgParse():
    """Get Args"""
    parser = argparse.ArgumentParser(
        description="Penalty-Free Neural Network Method")
    parser.add_argument("--problem", type=int, default=1,
                        help="the type of the problem")
    parser.add_argument("--bound", type=float, default=[-1.0, 1.0, -1.0, 1.0],
                        help="lower and upper bound of the domain")
    parser.add_argument("--inset_nx", type=int, default=[60, 60],
                        help="size of the inner set")
    parser.add_argument("--bdset_nx", type=int, default=[60, 60],
                        help="size of the boundary set")
    parser.add_argument("--teset_nx", type=int, default=[101, 101],
                        help="size of the test set")
    parser.add_argument("--g_epochs", type=int, default=6000,
                        help="number of epochs to train neural network g")
    parser.add_argument("--f_epochs", type=int, default=6000,
                        help="number of epochs to train neural network f")
    parser.add_argument("--g_lr", type=float, default=0.01,
                        help="learning rate to train neural network g")
    parser.add_argument("--f_lr", type=float, default=0.01,
                        help="learning rate to train neural network f")
    parser.add_argument("--use_cuda", type=bool, default=True,
                        help="device")
    parser.add_argument("--tests_num", type=int, default=5,
                        help="number of independent tests")
    parser.add_argument("--g_path", type=str, default="./optimal_state/optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netg")
    parser.add_argument("--f_path", type=str, default="./optimal_state/optimal_state_f_pfnn.ckpt",
                        help="the path that will put checkpoint of netf")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = ArgParse()
    if args.use_cuda:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, devide_target="CPU")
    errors = np.zeros(args.tests_num)
    for ii in range(args.tests_num):
        InSet, BdSet, TeSet = Data.GenerateSet(args)
        dsg, dsloss = DataSet.GenerateDataSet(InSet, BdSet)

        lenfac = model.LenFac(Tensor(args.bound, mstype.float32).reshape(2, 2), 1)
        netg = model.NetG()
        netf = model.NetF()
        netloss = model.Loss(netf)
        optimg = nn.Adam(netg.trainable_params(), learning_rate=args.g_lr)
        optimf = nn.Adam(netf.trainable_params(), learning_rate=args.f_lr)

        start_time = time.time()
        train.train(args, netg, netf, netloss, lenfac, optimg, optimf, InSet, BdSet, dsg, dsloss)
        elapsed = time.time() - start_time
        print("train time: %.2f" % (elapsed))

        load_param_into_net(netg, load_checkpoint(
            args.g_path), strict_load=True)
        load_param_into_net(netf, load_checkpoint(
            args.f_path), strict_load=True)
        errors[ii] = evalerror.EvalError(netg, netf, lenfac, TeSet)
        print("test_error = %.3e\n" % (errors[ii].item()))

    print(errors)
    errors_mean = errors.mean()
    errors_std = errors.std()

    print("test_error_mean = %.3e, test_error_std = %.3e"
          % (errors_mean.item(), errors_std.item()))
