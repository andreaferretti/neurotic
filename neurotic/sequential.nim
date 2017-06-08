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
import neo
import ./core

type
  Sequential*[A] = ref object of Layer[A]
    modules: seq[Layer[A]]

proc add*[A: SomeReal](m: var Sequential[A], layer: Layer[A]) = m.modules.add(layer)

method forward*[A: SomeReal](m: Sequential[A], x: Vector[A]): auto =
  m.modules.foldl(b.forward(a), x)

method forward*[A: SomeReal](m: Sequential[A], x: Matrix[A]): auto =
  m.modules.foldl(b.forward(a), x)

template backwardT(m, x, eta, result: untyped) =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

method backward*[A: SomeReal](m: Sequential[A], x: Vector[A], eta: float32): Vector[A] =
  backwardT(m, x, eta, result)

method backward*[A: SomeReal](m: Sequential[A], x: Matrix[A], eta: float32): Matrix[A] =
  backwardT(m, x, eta, result)

proc `->`*[A: SomeReal](a, b: Layer[A]): Sequential[A] = Sequential[A](modules: @[a, b])

proc sequential*[A: SomeReal](modules: seq[Layer[A]]): Sequential[A] =
  Sequential[A](modules: @modules)

method inputSize*[A](s: Sequential[A]): int =
  s.modules[0].inputSize

method outputSize*[A](s: Sequential[A]): int =
  s.modules[s.modules.high].outputSize