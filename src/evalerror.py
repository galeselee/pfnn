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
"""eval function"""
from mindspore import Tensor
from mindspore import dtype as mstype

def EvalError(netg, netf, lenfac, TeSet):
    """
    The eval function

    Args:
        netg: Instantiation of NetG
        netf: Instantiation of NetF
        lenfac： Instantiation of LenFac
        TeSet: Test Dataset

    Return:
        error: Test error
   """
    x = Tensor(TeSet.x, mstype.float32)
    TeSet_u = (netg(x)+lenfac(Tensor(x[:, 0])
                              ).reshape(-1, 1) * netf(x)).asnumpy()
    error = (((TeSet_u - TeSet.ua)**2).sum() / (TeSet.ua).sum()) ** 0.5
    return error
