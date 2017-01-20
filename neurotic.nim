import future, math
import linalg
import random/urandom, random/mersenne

type
  Module64 = concept m
    # var x: DMatrix64
    var y: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y) is DVector64
    # m.backward(x) is DMatrix64
    # m.backward(y) is DVector64
  Cost64 = concept m
    # var x: DMatrix64
    var y, t: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y, t) is float64
    # m.backward(x) is DMatrix64
    m.backward(y, t) is DVector64
  Layer64 = concept x
    x.withMemory is Module64
  Result64 = object
    loss: float64
    gradient: DVector64
  Dense64 = object
    a, b: int
  Dense64Memory = object
    weights: DMatrix64
    bias: DVector64
  Dense64Module = object
    memory: Dense64Memory
    lastInput: DVector64
  Sigmoid64 = object
    lastInput: DVector64
  Sequential[A, B] = object
    module1: A
    module2: B
  Closure64 = object
    fwd: DVector64 -> DVector64
    bkd: (DVector64, float64) -> DVector64
  QuadraticCost = object

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

proc forward(m: var Dense64Module, x: DVector64): DVector64 =
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

proc backward(m: var Dense64Module, v: DVector64, eta = 0.01'f64): DVector64 =
  result = m.memory.weights.t * v
  let gradWeights = v .* m.lastInput
  m.memory.bias -= eta * v
  m.memory.weights -= eta * gradWeights

proc forward[A, B: Module64](m: var Sequential[A, B], x: DVector64): DVector64 =
  m.module2.forward(m.module1.forward(x))

proc backward[A, B: Module64](m: var Sequential[A, B], x: DVector64, eta = 0.01'f64): DVector64 =
  m.module1.backward(m.module2.backward(x, eta), eta)

proc forward(m: var Closure64, x: DVector64): DVector64 = m.fwd(x)

proc backward(m: var Closure64, x: DVector64, eta = 0.01'f64): DVector64 = m.bkd(x, eta)

proc into[A: Module64; B: Module64](a: A, b: B): Closure64 =
  var a1 = a
  var b1 = b

  proc fwd(x: DVector64): DVector64 = b1.forward(a1.forward(x))

  proc bkd(x: DVector64, eta: float64): DVector64 =
    a1.backward(b1.backward(x, eta), eta)

  return Closure64(fwd: fwd, bkd: bkd)


proc `->`[A: Module64; B: Module64](a: var A, b: var B): auto =
  Sequential[A, B](module1: a, module2: b)

proc forward(m: QuadraticCost, x, y: DVector64): float64 = l_2(x - y)

proc backward(m: QuadraticCost, x, y: DVector64): DVector64 = 2 * (x - y)

proc sigmoid(z: float64): float64 = 1.0 / (exp(-z) + 1.0)

proc sigmoidPrime(z: float64): float64 = sigmoid(z) * (1.0 - sigmoid(z))

makeUniversal(sigmoid)
makeUniversal(sigmoidPrime)

proc forward(m: var Sigmoid64, x: DVector64): DVector64 =
  m.lastInput = x
  return sigmoid(x)

proc backward(m: var Sigmoid64, v: DVector64): DVector64 =
  sigmoidPrime(m.lastInput) |*| v

proc run(m: var Module64, c: Cost64, input, output: DVector64): Result64 =
  let
    prediction = m.forward(input)
    loss = c.forward(prediction, output)
    firstGradient = c.backward(prediction, output)
    gradient = m.backward(firstGradient)
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
    m4 = m1 -> m3
    # m5 = m3 -> m4
    # m5 = m4 -> m3
  let
    v = randomVector(784).toDynamic
    w = randomVector(20).toDynamic
  let result = run(m4, cost, v, w)
  echo result.gradient
  echo result.loss
  echo m1 is Module64
  echo m2 is Module64
  # echo((m1 -> m2) is Module64)
  # echo((m2 -> m3) is Module64)
  # echo((m1 -> m3) is Module64)
  echo l1 is Layer64
  echo m2 is Module64
  echo cost is Cost64