import linalg
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

var rng = initMersenneTwister(urandom(16))

proc dense64*(a, b: int): auto = Dense64(a: a, b: b)

proc dense*(a, b: int): auto = dense64(a, b)

proc memory*(d: Dense64): Dense64Memory =
  Dense64Memory(
    weights: makeMatrix(d.b, d.a, proc(i, j: int): float64 = rng.random()),
    bias: makeVector(d.b, proc(i: int): float64 = rng.random())
  )

proc withMemory*(d: Dense64, m: Dense64Memory): Dense64Module =
  Dense64Module(memory: m)

proc withMemory*(d: Dense64): Dense64Module = d.withMemory(d.memory)

proc forward*(m: Dense64Module, x: DVector64): DVector64 =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

proc backward*(m: Dense64Module, v: DVector64, eta: float64): DVector64 =
  result = m.memory.weights.t * v
  let gradWeights = v .* m.lastInput
  m.memory.bias -= eta * v
  m.memory.weights -= eta * gradWeights