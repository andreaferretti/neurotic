import linalg
import ./util

type
  Activation* = ref object of RootObj
    lastInput: DVector64
    f: proc(x: DVector64): DVector64
    fPrime: proc(x: DVector64): DVector64

proc sigmoid*(z: float64): float64 = 1.0 / (exp(-z) + 1.0)

proc sigmoidPrime*(z: float64): float64 = sigmoid(z) * (1.0 - sigmoid(z))

proc relu*(z: float64): float64 = max(z, 0.0)

proc reluPrime*(z: float64): float64 =
  if z >= 0: 1.0 else: 0.0

makeUniversal(sigmoid)
makeUniversal(sigmoidPrime)
makeUniversal(relu)
makeUniversal(reluPrime)

proc forward*(a: Activation, x: DVector64): DVector64 =
  a.lastInput = x
  return a.f(x)

proc backward*(a: Activation, v: DVector64, eta: float64): DVector64 =
  a.fPrime(a.lastInput) |*| v

proc sigmoidModule*(): Activation = Activation(f: sigmoid, fPrime: sigmoidPrime)

proc reluModule*(): Activation = Activation(f: relu, fPrime: reluPrime)