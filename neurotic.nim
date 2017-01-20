import future
import linalg
import random/urandom, random/mersenne

type
  Module64 = concept m
    var x: DMatrix64
    var y: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y) is DVector64
    # m.backward(x) is DMatrix64
    m.backward(y) is DVector64
  Cost64 = concept m
    var x: DMatrix64
    var y: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y) is float64
    # m.backward(x) is DMatrix64
    m.backward(y) is float64
  Dense64 = object
    a, b: int
  Dense64Memory = object
    weights: DMatrix64
    bias: DVector64
  Dense64Module = object
    memory: Dense64Memory
    lastInput: DVector64

var rng = initMersenneTwister(urandom(16))

proc dense64(a, b: int): auto = Dense64(a: a, b: b)

proc dense(a, b: int): auto = dense64(a, b)

proc memory(d: Dense64): Dense64Memory =
  Dense64Memory(
    weights: makeMatrix(d.b, d.a, proc(i, j: int): float64 = rng.random()),
    bias: makeVector(d.b, proc(i: int): float64 = rng.random())
  )

proc withMemory(d: Dense64, m: Dense64Memory): Dense64Module =
  Dense64Module(memory: m)

proc withMemory(d: Dense64): Dense64Module = d.withMemory(d.memory)

# proc forward(m: Dense64Module, x: DMatrix64): DMatrix64 = (m.weights * x) +

proc forward(m: var Dense64Module, x: DVector64): DVector64 =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

proc `.*`(a, b: DVector64): DMatrix64 =
  makeMatrix(a.len, b.len, proc(i, j: int): float64 = a[i] * b[j])

proc backward(m: var Dense64Module, v: DVector64, eta = 0.01'f64): DVector64 =
  result = m.memory.weights.t * v
  let gradWeights = v .* m.lastInput
  m.memory.bias -= eta * v
  m.memory.weights -= eta * gradWeights

when isMainModule:
  let l1 = dense(784, 30)
  var m1 = l1.withMemory
  let v = randomVector(784).toDynamic
  let output = m1.forward(v)
  echo output
  echo(output .* output)
  let x = m1.backward(output)
  echo x
  echo m1 is Module64