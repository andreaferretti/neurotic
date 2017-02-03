import math
import linalg, alea
import random/urandom, random/mersenne
import ./core, ./util

type
  Dense64* = object
    a, b: int
  Dense64Memory* = object
    weights*, gradWeights*: DMatrix64
    bias*, gradBias*: DVector64
  Dense64Layer = ref object of Layer64
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

proc withMemory*(d: Dense64, m: Dense64Memory): Dense64Layer =
  Dense64Layer(memory: m)

proc withMemory*(d: Dense64): Dense64Layer = d.withMemory(d.memory)

method forward*(m: Dense64Layer, x: DVector64): DVector64 =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

method forward*(m: Dense64Layer, x: DMatrix64): DMatrix64 =
  m.lastInputs = x
  let (_, n) = x.dim
  return (m.memory.weights * x) + repeat(m.memory.bias, n)

method backward*(m: Dense64Layer, v: DVector64, eta: float64): DVector64 =
  result = m.memory.weights.t * v
  m.memory.gradBias = v
  m.memory.gradWeights = v.asMatrix(v.len, 1) * m.lastInput.asMatrix(1, m.lastInput.len)#  v .* m.lastInput
  m.memory.bias -= eta * m.memory.gradBias
  m.memory.weights -= eta * m.memory.gradWeights

method backward*(m: Dense64Layer, v: DMatrix64, eta: float64): DMatrix64 =
  result = m.memory.weights.t * v
  let (_, n) = v.dim
  let k = n.float64
  m.memory.gradBias = sumColumns(v) / k
  m.memory.gradWeights = v * m.lastInputs.t / k
  # Shouldn't wee multiply by n again?
  m.memory.bias -= eta * m.memory.gradBias
  m.memory.weights -= eta * m.memory.gradWeights