import math
import linalg, alea
import random/urandom, random/mersenne
import ./core, ./util

type
  Dense64* = object
    a, b: int
  Dense64Memory* = object
    weights*: DMatrix64
    bias*: DVector64
  Dense64Module = ref object of RootObj
    memory*: Dense64Memory
    lastInput: DVector64
    lastInputs: DMatrix64

var rng = wrap(initMersenneTwister(urandom(16)))
let g = gaussian(mu = 0, sigma = 1)

proc dense64*(a, b: int): auto = Dense64(a: a, b: b)

proc dense*(a, b: int): auto = dense64(a, b)

proc memory*(d: Dense64): Dense64Memory =
  Dense64Memory(
    weights: makeMatrix(d.b, d.a, proc(i, j: int): float64 = rng.sample(g) / sqrt(d.a.float)),
    bias: makeVector(d.b, proc(i: int): float64 = rng.sample(g))
  )

proc withMemory*(d: Dense64, m: Dense64Memory): Dense64Module =
  Dense64Module(memory: m)

proc withMemory*(d: Dense64): Dense64Module = d.withMemory(d.memory)

proc forward*(m: Dense64Module, x: DVector64): DVector64 =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

proc forwardM*(m: Dense64Module, x: DMatrix64): DMatrix64 =
  m.lastInputs = x
  let (_, n) = x.dim
  return (m.memory.weights * x) + repeat(m.memory.bias, n)

proc backward*(m: Dense64Module, v: DVector64, eta: float64): DVector64 =
  result = m.memory.weights.t * v
  let gradWeights = v .* m.lastInput
  m.memory.bias -= eta * v
  m.memory.weights -= eta * gradWeights