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
"""Evaler"""
import argparse
import numpy as np
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore import load_param_into_net, load_checkpoint
from src import evalerror, model
from Data import Data


def ArgParser():
    """Get Args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, default=1,
                        help="the type of the problem")
    parser.add_argument("--bound", type=float, default=[-1.0, 1.0, -1.0, 1.0],
                        help="lower and upper bound of the domain")
    parser.add_argument("--teset_nx", type=int, default=[101, 101],
                        help="size of the test set")
    parser.add_argument("--use_cuda", type=bool, default=True,
                        help="device")
    parser.add_argument("--g_path", type=str, default="./optimal_state/optimal_state_g_pfnn.ckpt",
                        help="the path that will put checkpoint of netg")
    parser.add_argument("--f_path", type=str, default="./optimal_state/optimal_state_f_pfnn.ckpt",
                        help="the path that will put checkpoint of netf")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = ArgParser()
    if args.use_cuda:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, devide_target="CPU")

    bound = np.array(args.bound).reshape(2, 2)
    TeSet = Data.TestSet(bound, args.teset_nx, args.problem)
    lenfac = model.LenFac(Tensor(args.bound, mstype.float32).reshape(2, 2), 1)

    netg = model.NetG()
    netf = model.NetF()
    netloss = model.Loss(netf)
    load_param_into_net(netg, load_checkpoint(args.g_path), strict_load=True)
    load_param_into_net(netf, load_checkpoint(args.f_path), strict_load=True)
    error = evalerror.EvalError(netg, netf, lenfac, TeSet)
    print(f"The Test Error: {error}")
