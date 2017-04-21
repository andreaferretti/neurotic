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

type
  Cost32* = concept c
    var x, s: DMatrix32
    var y, t: DVector32
    c.forward(x, s) is float32
    c.forward(y, t) is float32
    c.backward(x, s) is DMatrix32
    c.backward(y, t) is DVector32
  Cost64* = concept c
    var x, s: DMatrix64
    var y, t: DVector64
    c.forward(x, s) is float64
    c.forward(y, t) is float64
    c.backward(x, s) is DMatrix64
    c.backward(y, t) is DVector64
  Layer32* = ref object of RootObj
  Layer64* = ref object of RootObj

method inputSize*(m: Layer32): int {.base.} =
  quit "to override!"

method inputSize*(m: Layer64): int {.base.} =
  quit "to override!"

method outputSize*(m: Layer32): int {.base.} =
  quit "to override!"

method outputSize*(m: Layer64): int {.base.} =
  quit "to override!"

method forward*(m: Layer32, v: DVector32): DVector32 {.base.} =
  quit "to override!"

method forward*(m: Layer32, v: DMatrix32): DMatrix32 {.base.} =
  quit "to override!"

method backward*(m: Layer32, v: DVector32, eta: float32): DVector32 {.base.} =
  quit "to override!"

method backward*(m: Layer32, v: DMatrix32, eta: float32): DMatrix32 {.base.} =
  quit "to override!"

method forward*(m: Layer64, v: DVector64): DVector64 {.base.} =
  quit "to override!"

method forward*(m: Layer64, v: DMatrix64): DMatrix64 {.base.} =
  quit "to override!"

method backward*(m: Layer64, v: DVector64, eta: float64): DVector64 {.base.} =
  quit "to override!"

method backward*(m: Layer64, v: DMatrix64, eta: float64): DMatrix64 {.base.} =
  quit "to override!"