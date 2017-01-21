import sequtils
import linalg
import ./core

type
  Sequential* = ref object of RootObj
    modules: seq[Module64]

proc forward*(m: Sequential, x: DVector64): DVector64 =
  m.modules.foldl(b.forward(a), x)

proc backward*(m: Sequential, x: DVector64, eta: float64): DVector64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

proc `->`*[A; B](a: A, b: B): Sequential =
  Sequential(modules: @[a.asModule64, b.asModule64])