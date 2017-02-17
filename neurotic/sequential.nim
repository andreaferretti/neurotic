import sequtils, macros
import linalg
import ./core

type
  Sequential32* = ref object of Layer32
    modules: seq[Layer32]
  Sequential64* = ref object of Layer64
    modules: seq[Layer64]

proc add*(m: var Sequential32, layer: Layer32) = m.modules.add(layer)

proc add*(m: var Sequential64, layer: Layer64) = m.modules.add(layer)

method forward*(m: Sequential32, x: DVector32): auto =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential32, x: DMatrix32): auto =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential64, x: DVector64): auto =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential64, x: DMatrix64): auto =
  m.modules.foldl(b.forward(a), x)

template backwardT(m, x, eta, result: untyped) =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

method backward*(m: Sequential32, x: DVector32, eta: float32): DVector32 =
  backwardT(m, x, eta, result)

method backward*(m: Sequential64, x: DVector64, eta: float64): DVector64 =
  backwardT(m, x, eta, result)

method backward*(m: Sequential32, x: DMatrix32, eta: float32): DMatrix32 =
  backwardT(m, x, eta, result)

method backward*(m: Sequential64, x: DMatrix64, eta: float64): DMatrix64 =
  backwardT(m, x, eta, result)

proc `->`*(a, b: Layer32): Sequential32 = Sequential32(modules: @[a, b])

proc `->`*(a, b: Layer64): Sequential64 = Sequential64(modules: @[a, b])

proc sequential*(modules: seq[Layer32]): Sequential32 =
  Sequential32(modules: @modules)

proc sequential*(modules: seq[Layer64]): Sequential64 =
  Sequential64(modules: @modules)