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

proc batch*(data: seq[TrainingData64], start, size: int): tuple[input, output: DMatrix64] =
  let
    inputSize = data[0].input.len
    outputSize = data[0].output.len
  var
    input = zeros(inputSize, size)
    output = zeros(outputSize, size)
  for i in 0 ..< size:
    let d = data[start + i]
    for j in 0 ..< inputSize:
      input[j, i] = d.input[j]
    for j in 0 ..< outputSize:
      output[j, i] = d.output[j]
  return (input, output)

proc miniBatchSgd*(m: Layer64, c: Cost64, data: seq[TrainingData64], batchSize = 100, eta = 0.01'f64) =
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