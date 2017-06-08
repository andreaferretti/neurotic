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

type
  Cost*[A] = concept c
    var x, s: Matrix[A]
    var y, t: Vector[A]
    c.forward(x, s) is A
    c.forward(y, t) is A
    c.backward(x, s) is Matrix[A]
    c.backward(y, t) is Vector[A]
  Layer*[A] = ref object of RootObj

method inputSize*[A](m: Layer[A]): int {.base.} =
  quit "to override!"

method outputSize*[A](m: Layer[A]): int {.base.} =
  quit "to override!"

method forward*[A: SomeReal](m: Layer[A], v: Vector[A]): Vector[A] {.base.} =
  quit "to override!"

method forward*[A: SomeReal](m: Layer[A], v: Matrix[A]): Matrix[A] {.base.} =
  quit "to override!"

method backward*[A: SomeReal](m: Layer[A], v: Vector[A], eta: A): Vector[A] {.base.} =
  quit "to override!"

method backward*[A: SomeReal](m: Layer[A], v: Matrix[A], eta: A): Matrix[A] {.base.} =
  quit "to override!"