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
import ./core, ./cost

type
  Result*[A] = object
    loss*: A
  TrainingData*[A] = tuple
    input, output: Vector[A]

template runT(m, c, input, output, eta, result: untyped) =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  result.loss = loss

proc run*[A: SomeReal](m: Layer[A], c: Cost[A], input, output: Vector[A], eta: A = 0.01): Result[A] =
  runT(m, c, input, output, eta, result)

proc run*[A: SomeReal](m: Layer[A], c: Cost[A], input, output: Matrix[A], eta: A = 0.01): Result[A] =
  runT(m, c, input, output, eta, result)

proc sgd*[A: SomeReal](m: Layer[A], c: Cost[A], data: seq[TrainingData[A]], eta: A = 0.01) =
  var
    count = 0
    loss = 0.0
  for d in data:
    let
      (input, output) = d
      res = run(m, c, input, output, eta)
    loss += res.loss
    count += 1
  echo "loss: ", (loss / count.float)


proc batch*[A: SomeReal](data: seq[TrainingData[A]], start, size: int): tuple[input, output: Matrix[A]] =
  let
    inputSize = data[0].input.len
    outputSize = data[0].output.len
  block:
    result.input = makeMatrixIJ(A, inputSize, size, data[start + j].input[i])
  block:
    result.output = makeMatrixIJ(A, outputSize, size, data[start + j].output[i])


proc miniBatchSgd*[A: SomeReal](m: Layer[A], c: Cost[A], data: seq[TrainingData[A]], batchSize = 100, eta: A = 0.01) =
  var
    count = 0
    loss = 0.0
  while count < data.len:
    let
      (input, output) = data.batch(count, batchSize)
      res = run(m, c, input, output, eta)
    loss += res.loss
    count += batchSize
  echo "loss: ", (loss / count.float)