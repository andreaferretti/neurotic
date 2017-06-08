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

import math
import neo, alea
import random/urandom, random/mersenne
import ./core, ./util

type
  Dense*[A] = object
    a, b: int
  DenseMemory*[A] = object
    weights*, gradWeights*: Matrix[A]
    bias*, gradBias*: Vector[A]
  DenseLayer*[A] = ref object of Layer[A]
    a, b: int
    memory*: DenseMemory[A]
    lastInput: Vector[A]
    lastInputs: Matrix[A]

var rng = wrap(initMersenneTwister(urandom(16)))
let g = gaussian(mu = 0, sigma = 1)

proc dense32*(a, b: int): auto = Dense[float32](a: a, b: b)

proc dense64*(a, b: int): auto = Dense[float64](a: a, b: b)

proc dense*(a, b: int): auto = dense64(a, b)

proc memory*[A: SomeReal](d: Dense[A]): DenseMemory[A] =
  DenseMemory[A](
    weights: makeMatrixIJ(A, d.b, d.a, A(rng.sample(g)) / sqrt(A(d.a))),
    bias: makeVectorI[A](d.b, A(rng.sample(g)))
  )

proc withMemory*[A: SomeReal](d: Dense[A], m: DenseMemory[A]): DenseLayer[A] =
  DenseLayer[A](a: d.a, b: d.b, memory: m)

proc withMemory*[A: SomeReal](d: Dense[A]): auto = d.withMemory(d.memory)

method forward*[A: SomeReal](m: DenseLayer[A], x: Vector[A]): Vector[A] =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

method forward*[A: SomeReal](m: DenseLayer[A], x: Matrix[A]): Matrix[A] =
  m.lastInputs = x
  let (_, n) = x.dim
  return (m.memory.weights * x) + repeat(m.memory.bias, n)

method backward*[A: SomeReal](m: DenseLayer[A], x: Vector[A], eta: A): Vector[A] =
  result = m.memory.weights.t * x
  m.memory.gradBias = x
  m.memory.gradWeights = x.asMatrix(x.len, 1) * m.lastInput.asMatrix(1, m.lastInput.len)
  m.memory.bias -= eta * m.memory.gradBias
  m.memory.weights -= eta * m.memory.gradWeights

method backward*[A: SomeReal](m: DenseLayer[A], x: Matrix[A], eta: A): Matrix[A] =
  result = m.memory.weights.t * x
  let k = A(x.N)
  m.memory.gradBias = sumColumns(x) / k
  m.memory.gradWeights = x * m.lastInputs.t / k
  # Shouldn't wee multiply by n again?
  m.memory.bias -= eta * m.memory.gradBias
  m.memory.weights -= eta * m.memory.gradWeights

method inputSize*[A: SomeReal](m: DenseLayer[A]): int = m.a

method outputSize*[A: SomeReal](m: DenseLayer[A]): int = m.b