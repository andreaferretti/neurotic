import linalg
import ./core, ./cost

type Result64* = object
  loss*: float64
  gradient*: DVector64

proc run*(m: IModule64, c: Cost64, input, output: DVector64, eta = 0.01'f64): Result64 =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  return Result64(loss: loss, gradient: gradient)