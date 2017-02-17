import math
import linalg, alea
import random/urandom, random/mersenne
import ./core, ./util

type
  Dense32* = object
    a, b: int
  Dense64* = object
    a, b: int
  Dense32Memory* = object
    weights*, gradWeights*: DMatrix32
    bias*, gradBias*: DVector32
  Dense64Memory* = object
    weights*, gradWeights*: DMatrix64
    bias*, gradBias*: DVector64
  Dense32Layer* = ref object of Layer32
    memory*: Dense32Memory
    lastInput: DVector32
    lastInputs: DMatrix32
  Dense64Layer* = ref object of Layer64
    memory*: Dense64Memory
    lastInput: DVector64
    lastInputs: DMatrix64

var rng = wrap(initMersenneTwister(urandom(16)))
let g = gaussian(mu = 0, sigma = 1)

proc dense32*(a, b: int): auto = Dense32(a: a, b: b)

proc dense64*(a, b: int): auto = Dense64(a: a, b: b)

proc dense*(a, b: int): auto = dense64(a, b)

proc memory*(d: Dense32): Dense32Memory =
  Dense32Memory(
    weights: makeMatrix(d.b, d.a, proc(i, j: int): float32 = rng.sample(g) / sqrt(d.a.float32)),
    bias: makeVector(d.b, proc(i: int): float32 = rng.sample(g))
  )

proc memory*(d: Dense64): Dense64Memory =
  Dense64Memory(
    weights: makeMatrix(d.b, d.a, proc(i, j: int): float64 = rng.sample(g) / sqrt(d.a.float)),
    bias: makeVector(d.b, proc(i: int): float64 = rng.sample(g))
  )

proc withMemory*(d: Dense32, m: Dense32Memory): Dense32Layer =
  Dense32Layer(memory: m)

proc withMemory*(d: Dense64, m: Dense64Memory): Dense64Layer =
  Dense64Layer(memory: m)

proc withMemory*(d: Dense32 or Dense64): auto = d.withMemory(d.memory)

template forwardVector(m, x: untyped) =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

method forward*(m: Dense32Layer, x: DVector32): DVector32 = forwardVector(m, x)

method forward*(m: Dense64Layer, x: DVector64): DVector64 = forwardVector(m, x)

template forwardMatrix(m, x: untyped) =
  m.lastInputs = x
  let (_, n) = x.dim
  return (m.memory.weights * x) + repeat(m.memory.bias, n)

method forward*(m: Dense32Layer, x: DMatrix32): DMatrix32 = forwardMatrix(m, x)

method forward*(m: Dense64Layer, x: DMatrix64): DMatrix64 = forwardMatrix(m, x)

template backwardVector(m, x, eta: untyped) =
  result = m.memory.weights.t * x
  m.memory.gradBias = x
  m.memory.gradWeights = x.asMatrix(x.len, 1) * m.lastInput.asMatrix(1, m.lastInput.len)
  m.memory.bias -= eta * m.memory.gradBias
  m.memory.weights -= eta * m.memory.gradWeights

method backward*(m: Dense32Layer, x: DVector32, eta: float32): DVector32 =
  backwardVector(m, x, eta)

method backward*(m: Dense64Layer, x: DVector64, eta: float64): DVector64 =
  backwardVector(m, x, eta)

template backwardMatrix(m, x, eta: untyped) =
  result = m.memory.weights.t * x
  let (_, n) = x.dim
  when x is DMatrix32:
    let k = n.float32
  else:
    let k = n.float64
  m.memory.gradBias = sumColumns(x) / k
  m.memory.gradWeights = x * m.lastInputs.t / k
  # Shouldn't wee multiply by n again?
  m.memory.bias -= eta * m.memory.gradBias
  m.memory.weights -= eta * m.memory.gradWeights

method backward*(m: Dense32Layer, x: DMatrix32, eta: float32): DMatrix32 =
  backwardMatrix(m, x, eta)

method backward*(m: Dense64Layer, x: DMatrix64, eta: float64): DMatrix64 =
  backwardMatrix(m, x, eta)