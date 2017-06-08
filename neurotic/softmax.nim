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

import neo
import ./core, ./util

type
  SoftMax*[A] = ref object of Layer[A]
    lastOutput: Vector[A]
    lastOutputs: Matrix[A]

proc exp1(z: float32 or float64): auto = exp(z)

makeUniversal(exp1)

proc softMax*(A: typedesc): SoftMax[A] =
  new result

proc softMax*[A: SomeReal](v: Vector[A]): auto =
  let y = exp1(v)
  return y / l_1(y)

proc softMax*[A: SomeReal](m: Matrix[A]): auto =
  result = exp1(m)
  let (a, b) = m.dim
  var v = zeros(b)
  for id, x in result:
    let (_, j) = id
    v[j] += x
  for i in 0 ..< a:
    for j in 0 ..< b:
      result[i, j] = result[i, j] / v[j]

method forward*[A: SomeReal](a: SoftMax[A], x: Vector[A]): Vector[A] =
  result = softMax(x)
  a.lastOutput = result

method forward*[A: SomeReal](a: SoftMax[A], x: Matrix[A]): Matrix[A] =
  result = softMax(x)
  a.lastOutputs = result

method backward*[A: SomeReal](a: SoftMax[A], v: Vector[A], eta: A): Vector[A] =
  # let jacobian = a.lastOutput.diagonal - (a.lastOutput.vertical * a.lastOutput.horizontal)
  # return jacobian * v
  (a.lastOutput |*| v) - ((a.lastOutput * v) * a.lastOutput)

method backward*[A: SomeReal](a: SoftMax[A], v: Matrix[A], eta: A): Matrix[A] =
  (a.lastOutputs |*| v) - (a.lastOutputs * (a.lastOutputs.t * v))