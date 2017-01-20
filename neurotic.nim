import future, math, sequtils
import linalg
import random/urandom, random/mersenne
import collections/iface

type
  # Module64 = concept m
  #   # var x: DMatrix64
  #   var y: DVector64
  #   # m.forward(x) is DMatrix64
  #   m.forward(y) is DVector64
  #   # m.backward(x) is DMatrix64
  #   m.backward(y) is DVector64
  Cost64 = concept m
    # var x: DMatrix64
    var y, t: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y, t) is float64
    # m.backward(x) is DMatrix64
    m.backward(y, t) is DVector64
  Layer64 = concept x
    x.withMemory is Module64
  IModule64 = distinct Interface
  Result64 = object
    loss: float64
    gradient: DVector64
  Dense64 = object
    a, b: int
  Dense64Memory = object
    weights: DMatrix64
    bias: DVector64
  Dense64Module = ref object of RootObj
    memory: Dense64Memory
    lastInput: DVector64
  Sigmoid64 = ref object of RootObj
    lastInput: DVector64
  Sequential1 = ref object of RootObj
    modules: seq[IModule64]
  QuadraticCost = object

interfaceMethods IModule64:
  forward(v: DVector64): DVector64
  backward(v: DVector64, eta: float64): DVector64

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

proc forward(m: Dense64Module, x: DVector64): DVector64 =
  m.lastInput = x
  return (m.memory.weights * x) + m.memory.bias

# External (tensor) product
proc `.*`(a, b: DVector64): DMatrix64 =
  makeMatrix(a.len, b.len, proc(i, j: int): float64 = a[i] * b[j])

# Hadamard (component-wise) product
proc `|*|`(a, b: DVector64): DVector64 =
  assert a.len == b.len
  result = newSeq[float64](a.len)
  for i in 0 .. < a.len:
    result[i] = a[i] * b[i]

proc backward(m: Dense64Module, v: DVector64, eta: float64): DVector64 =
  result = m.memory.weights.t * v
  let gradWeights = v .* m.lastInput
  m.memory.bias -= eta * v
  m.memory.weights -= eta * gradWeights

proc forward(m: Sequential1, x: DVector64): DVector64 =
  m.modules.foldl(b.forward(a), x)

proc backward(m: Sequential1, x: DVector64, eta: float64): DVector64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

proc `->`[A; B](a: A, b: B): Sequential1 =
  Sequential1(modules: @[a.asIModule64, b.asIModule64])

proc forward(m: QuadraticCost, x, y: DVector64): float64 = l_2(x - y)

proc backward(m: QuadraticCost, x, y: DVector64): DVector64 = 2 * (x - y)

proc sigmoid(z: float64): float64 = 1.0 / (exp(-z) + 1.0)

proc sigmoidPrime(z: float64): float64 = sigmoid(z) * (1.0 - sigmoid(z))

makeUniversal(sigmoid)
makeUniversal(sigmoidPrime)

proc forward(m: Sigmoid64, x: DVector64): DVector64 =
  m.lastInput = x
  return sigmoid(x)

proc backward(m: Sigmoid64, v: DVector64, eta: float64): DVector64 =
  sigmoidPrime(m.lastInput) |*| v

proc run(m: IModule64, c: Cost64, input, output: DVector64, eta = 0.01'f64): Result64 =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient, eta)
  return Result64(loss: loss, gradient: gradient)


when isMainModule:
  let
    l1 = dense(784, 30)
    l2 = dense(30, 20)
    cost = QuadraticCost()
  var
    m1 = l1.withMemory
    m2 = Sigmoid64()
    m3 = l2.withMemory
    m4 =(m1 -> m2) -> m3
  let
    v = randomVector(784).toDynamic
    w = randomVector(20).toDynamic

  let result = run(m4.asIModule64, cost, v, w)
  echo result.gradient
  echo result.loss