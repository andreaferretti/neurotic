import sequtils, macros
import linalg
import ./core

type Sequential* = ref object of Module64
  modules: seq[Module64]

method forward*(m: Sequential, x: DVector64): DVector64 =
  m.modules.foldl(b.forward(a), x)

method forward*(m: Sequential, x: DMatrix64): DMatrix64 =
  m.modules.foldl(b.forward(a), x)

method backward*(m: Sequential, x: DVector64, eta: float64): DVector64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

method backward*(m: Sequential, x: DMatrix64, eta: float64): DMatrix64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

proc `->`*(a, b: Module64): Sequential =
  Sequential(modules: @[a, b])

proc sequential*(modules: seq[Module64]): Sequential =
  Sequential(modules: @modules)