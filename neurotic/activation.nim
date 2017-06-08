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
  Activation*[A] = ref object of Layer[A]
    lastInput: Vector[A]
    lastInputs: Matrix[A]
    f: proc(x: Vector[A]): Vector[A]
    fm: proc(x: Matrix[A]): Matrix[A]
    fPrime: proc(x: Vector[A]): Vector[A]
    fmPrime: proc(x: Matrix[A]): Matrix[A]

proc sigmoid*(z: float32 or float64): auto = 1.0 / (exp(-z) + 1.0)

proc sigmoidPrime*(z: float32 or float64): auto = sigmoid(z) * (1.0 - sigmoid(z))

proc relu*(z: float32 or float64): auto = max(z, 0.0)

proc reluPrime*(z: float32 or float64): auto =
  if z >= 0: 1.0 else: 0.0

makeUniversal(sigmoid)
makeUniversal(sigmoidPrime)
makeUniversal(relu)
makeUniversal(reluPrime)

method forward*[A: SomeReal](a: Activation[A], x: Vector[A]): Vector[A] =
  a.lastInput = x
  return a.f(x)

method forward*[A: SomeReal](a: Activation[A], x: Matrix[A]): Matrix[A] =
  a.lastInputs = x
  return a.fm(x)

method backward*[A: SomeReal](a: Activation[A], v: Vector[A], eta: A): Vector[A] =
  a.fPrime(a.lastInput) |*| v

method backward*[A: SomeReal](a: Activation[A], v: Matrix[A], eta: A): Matrix[A] =
  a.fmPrime(a.lastInputs) |*| v

proc sigmoidModule*(A: typedesc): Activation[A] = Activation[A](
  f: sigmoid,
  fm: sigmoid,
  fPrime: sigmoidPrime,
  fmPrime: sigmoidPrime
)

proc reluModule*(A: typedesc): Activation[A] = Activation[A](
  f: relu,
  fm: relu,
  fPrime: reluPrime,
  fmPrime: reluPrime
)