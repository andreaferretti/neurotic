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
import ./core, ./util

type
  Join32* = ref object of Layer32
    modules: seq[Layer32]
  Join64* = ref object of Layer64
    modules: seq[Layer64]

proc add*(m: var Join32, layer: Layer32) = m.modules.add(layer)

proc add*(m: var Join64, layer: Layer64) = m.modules.add(layer)

template forwardT(m, x, result: untyped) =
  let L = m.modules.len
  var
    count = 0
    outputs = newSeq[type(x)](L)
  for i in 0 ..< L:
    let
      size = m.modules[i].inputSize
      input = x[count ..< count + size]
    outputs[i] = m.modules[i].forward(input)
    count += size
  assert count == x.len
  result = concat(outputs)

method forward*(m: Join32, x: DVector32): auto =
  forwardT(m, x, result)

method forward*(m: Join32, x: DMatrix32): auto =
  quit "to write"
  # forwardT(m, x, result)

method forward*(m: Join64, x: DVector64): auto =
  forwardT(m, x, result)

method forward*(m: Join64, x: DMatrix64): auto =
  quit "to write"
  # forwardT(m, x, result)

template backwardT(m, x, eta, result: untyped) =
  quit "to write"

method backward*(m: Join32, x: DVector32, eta: float32): DVector32 =
  backwardT(m, x, eta, result)

method backward*(m: Join64, x: DVector64, eta: float64): DVector64 =
  backwardT(m, x, eta, result)

method backward*(m: Join32, x: DMatrix32, eta: float32): DMatrix32 =
  backwardT(m, x, eta, result)

method backward*(m: Join64, x: DMatrix64, eta: float64): DMatrix64 =
  backwardT(m, x, eta, result)

proc `+`*(a, b: Layer32): Join32 = Join32(modules: @[a, b])

proc `+`*(a, b: Layer64): Join64 = Join64(modules: @[a, b])

proc join*(modules: seq[Layer32]): Join32 =
  Join32(modules: @modules)

proc join*(modules: seq[Layer64]): Join64 =
  Join64(modules: @modules)