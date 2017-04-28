# Copyright 2017 UniCredit S.p.A.
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

import linalg
import ./core, ./util

type
  SoftMax32* = ref object of Layer32
    lastOutput: DVector32
    lastOutputs: DMatrix32
  SoftMax64* = ref object of Layer64
    lastOutput: DVector64
    lastOutputs: DMatrix64

proc exp1(z: float32 or float64): auto = exp(z)

makeUniversal(exp1)

proc softMax32*(): SoftMax32 =
  new result

proc softMax*(): SoftMax64 =
  new result

proc softMax*(v: DVector32 or DVector64): auto =
  let y = exp1(v)
  return y / l_1(y)

proc softMax*(m: DMatrix32 or DMatrix64): auto =
  result = exp1(m)
  let (a, b) = m.dim
  var v = zeros(b)
  for id, x in result:
    let (_, j) = id
    v[j] += x
  for i in 0 ..< a:
    for j in 0 ..< b:
      result[i, j] = result[i, j] / v[j]

method forward*(a: SoftMax32, x: DVector32): DVector32 =
  result = softMax(x)
  a.lastOutput = result

method forward*(a: SoftMax32, x: DMatrix32): DMatrix32 =
  result = softMax(x)
  a.lastOutputs = result

method backward*(a: SoftMax32, v: DVector32, eta: float32): DVector32 =
  # let jacobian = a.lastOutput.diagonal - (a.lastOutput.vertical * a.lastOutput.horizontal)
  # return jacobian * v
  (a.lastOutput |*| v) - ((a.lastOutput * v) * a.lastOutput)

method backward*(a: SoftMax32, v: DMatrix32, eta: float32): DMatrix32 =
  (a.lastOutputs |*| v) - (a.lastOutputs * (a.lastOutputs.t * v))

method forward*(a: SoftMax64, x: DVector64): DVector64 =
  result = softMax(x)
  a.lastOutput = result

method forward*(a: SoftMax64, x: DMatrix64): DMatrix64 =
  result = softMax(x)
  a.lastOutputs = result

method backward*(a: SoftMax64, v: DVector64, eta: float64): DVector64 =
  # let jacobian = a.lastOutput.diagonal - (a.lastOutput.vertical * a.lastOutput.horizontal)
  # return jacobian * v
  (a.lastOutput |*| v) - ((a.lastOutput * v) * a.lastOutput)

method backward*(a: SoftMax64, v: DMatrix64, eta: float64): DMatrix64 =
  (a.lastOutputs |*| v) - (a.lastOutputs * (a.lastOutputs.t * v))