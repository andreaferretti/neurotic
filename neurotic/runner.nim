import linalg
import ./core, ./cost

type
  Result64* = object
    loss*: float64
    gradient*: DVector64
  TrainingData64* = tuple
    input, output: DVector64

proc run*(m: Module64, c: Cost64, input, output: DVector64, eta = 0.01'f64): Result64 =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  return Result64(loss: loss, gradient: gradient)

proc sgd*(m: Module64, c: Cost64, data: seq[TrainingData64], eta = 0.01'f64) =
  var count = 0
  for d in data:
    let
      (input, output) = d
      res = run(m, c, input, output, eta)
    if count mod 1000 == 0:
      echo res.loss
    count += 1