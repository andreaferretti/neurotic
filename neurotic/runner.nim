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
import ./core, ./cost

type
  Result32* = object
    loss*: float32
  Result64* = object
    loss*: float64
  TrainingData32* = tuple
    input, output: DVector32
  TrainingData64* = tuple
    input, output: DVector64

template runT(m, c, input, output, eta, result: untyped) =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  result.loss = loss

proc run*(m: Layer32, c: Cost32, input, output: DVector32, eta = 0.01'f32): Result32 =
  runT(m, c, input, output, eta, result)

proc run*(m: Layer64, c: Cost64, input, output: DVector64, eta = 0.01'f64): Result64 =
  runT(m, c, input, output, eta, result)

proc run*(m: Layer32, c: Cost32, input, output: DMatrix32, eta = 0.01'f32): Result32 =
  runT(m, c, input, output, eta, result)

proc run*(m: Layer64, c: Cost64, input, output: DMatrix64, eta = 0.01'f64): Result64 =
  runT(m, c, input, output, eta, result)

template sgdT(m, c, data, eta: untyped) =
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

proc sgd*(m: Layer32, c: Cost32, data: seq[TrainingData32], eta = 0.01'f32) =
  sgdT(m, c, data, eta)

proc sgd*(m: Layer64, c: Cost64, data: seq[TrainingData64], eta = 0.01'f64) =
  sgdT(m, c, data, eta)

template batchT(data, start, size, result: untyped) =
  let
    inputSize = data[0].input.len
    outputSize = data[0].output.len
  block:
    result.input = makeMatrixIJD(inputSize, size, data[start + j].input[i])
  block:
    result.output = makeMatrixIJD(outputSize, size, data[start + j].output[i])

proc batch*(data: seq[TrainingData32], start, size: int): tuple[input, output: DMatrix32] =
  batchT(data, start, size, result)

proc batch*(data: seq[TrainingData64], start, size: int): tuple[input, output: DMatrix64] =
  batchT(data, start, size, result)

template miniBatchSgdT(m, c, data, batchSize, eta: untyped) =
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

proc miniBatchSgd*(m: Layer32, c: Cost32, data: seq[TrainingData32], batchSize = 100, eta = 0.01'f32) =
  miniBatchSgdT(m, c, data, batchSize, eta)

proc miniBatchSgd*(m: Layer64, c: Cost64, data: seq[TrainingData64], batchSize = 100, eta = 0.01'f64) =
  miniBatchSgdT(m, c, data, batchSize, eta)