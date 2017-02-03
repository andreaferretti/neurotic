import sequtils, macros
import linalg
import ./core

type Sequential64* = ref object of Layer64
  modules: seq[Layer64]

method forward*(m: Sequential64, x: DVector64): DVector64 =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential64, x: DMatrix64): DMatrix64 =
  m.modules.foldl(b.forward(a), x)

method backward*(m: Sequential64, x: DVector64, eta: float64): DVector64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

method backward*(m: Sequential64, x: DMatrix64, eta: float64): DMatrix64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

proc `->`*(a, b: Layer64): Sequential64 =
  Sequential64(modules: @[a, b])

proc sequential*(modules: seq[Layer64]): Sequential64 =
  Sequential64(modules: @modules)