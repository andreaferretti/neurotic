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

import sequtils, macros
import linalg
import ./core

type
  Sequential32* = ref object of Layer32
    modules: seq[Layer32]
  Sequential64* = ref object of Layer64
    modules: seq[Layer64]

proc add*(m: var Sequential32, layer: Layer32) = m.modules.add(layer)

proc add*(m: var Sequential64, layer: Layer64) = m.modules.add(layer)

method forward*(m: Sequential32, x: DVector32): auto =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential32, x: DMatrix32): auto =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential64, x: DVector64): auto =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential64, x: DMatrix64): auto =
  m.modules.foldl(b.forward(a), x)

template backwardT(m, x, eta, result: untyped) =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

method backward*(m: Sequential32, x: DVector32, eta: float32): DVector32 =
  backwardT(m, x, eta, result)

method backward*(m: Sequential64, x: DVector64, eta: float64): DVector64 =
  backwardT(m, x, eta, result)

method backward*(m: Sequential32, x: DMatrix32, eta: float32): DMatrix32 =
  backwardT(m, x, eta, result)

method backward*(m: Sequential64, x: DMatrix64, eta: float64): DMatrix64 =
  backwardT(m, x, eta, result)

proc `->`*(a, b: Layer32): Sequential32 = Sequential32(modules: @[a, b])

proc `->`*(a, b: Layer64): Sequential64 = Sequential64(modules: @[a, b])

proc sequential*(modules: seq[Layer32]): Sequential32 =
  Sequential32(modules: @modules)

proc sequential*(modules: seq[Layer64]): Sequential64 =
  Sequential64(modules: @modules)

method inputSize*(s: Sequential64): int =
  s.modules[0].inputSize

method outputSize*(s: Sequential64): int =
  s.modules[s.modules.high].outputSize