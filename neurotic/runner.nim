import linalg
import ./core, ./cost

type
  Result64* = object
    loss*: float64
  TrainingData64* = tuple
    input, output: DVector64

proc run*(m: Module64, c: Cost64, input, output: DVector64, eta = 0.01'f64): Result64 =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  return Result64(loss: loss)

proc run*(m: Module64, c: Cost64, input, output: DMatrix64, eta = 0.01'f64): Result64 =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  return Result64(loss: loss)

proc sgd*(m: Module64, c: Cost64, data: seq[TrainingData64], eta = 0.01'f64) =
  var count = 0
  for d in data:
    let
      (input, output) = d
      res = run(m, c, input, output, eta)
    if count mod 1000 == 0:
      echo res.loss
    count += 1

proc batch*(data: seq[TrainingData64], start, size: int): tuple[input, output: DMatrix64] =
  let
    inputSize = data[0].input.len
    outputSize = data[0].output.len
    input = makeMatrix(inputSize, size, proc(i, j: int): float64 = data[start + j].input[i])
    output = makeMatrix(outputSize, size, proc(i, j: int): float64 = data[start + j].output[i])
  return (input, output)

proc miniBatchSgd*(m: Module64, c: Cost64, data: seq[TrainingData64], batchSize = 100, eta = 0.01'f64) =
  var count = 0
  while count < data.len:
    let
      (input, output) = data.batch(count, batchSize)
      res = run(m, c, input, output, eta)
    if count mod 1000 == 0:
      echo res.loss
    count += batchSize