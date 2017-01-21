import linalg
import ./util

type
  Sigmoid64* = ref object of RootObj
    lastInput: DVector64
  Relu64* = ref object of RootObj
    lastInput: DVector64

proc sigmoid*(z: float64): float64 = 1.0 / (exp(-z) + 1.0)

proc sigmoidPrime*(z: float64): float64 = sigmoid(z) * (1.0 - sigmoid(z))

proc relu*(z: float64): float64 = max(z, 0.0)

proc reluPrime*(z: float64): float64 =
  if z >= 0: 1.0 else: 0.0

makeUniversal(sigmoid)
makeUniversal(sigmoidPrime)
makeUniversal(relu)
makeUniversal(reluPrime)

proc forward*(m: Sigmoid64, x: DVector64): DVector64 =
  m.lastInput = x
  return sigmoid(x)

proc backward*(m: Sigmoid64, v: DVector64, eta: float64): DVector64 =
  sigmoidPrime(m.lastInput) |*| v

proc forward*(m: Relu64, x: DVector64): DVector64 =
  m.lastInput = x
  return relu(x)

proc backward*(m: Relu64, v: DVector64, eta: float64): DVector64 =
  reluPrime(m.lastInput) |*| v