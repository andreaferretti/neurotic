import sequtils, macros
import linalg
import ./core

type Sequential* = ref object of RootObj
  modules: seq[Module64]

proc forward*(m: Sequential, x: DVector64): DVector64 =
  m.modules.foldl(b.forward(a), x)

proc backward*(m: Sequential, x: DVector64, eta: float64): DVector64 =
  result = x
  for i in countdown(m.modules.high, m.modules.low):
    result = m.modules[i].backward(result, eta)

proc `->`*[A; B](a: A, b: B): Sequential =
  Sequential(modules: @[a.asModule64, b.asModule64])

macro sequential*(modules: varargs[untyped]): auto =
  let asModule64Node = ident("asModule64")
  var b = newNimNode(nnkBracket)

  for m in modules:
    b.add(newDotExpr(m, asModule64Node))

  template inner(nodes: untyped): Sequential =
    Sequential(modules: @nodes)

  result = getAst(inner(b))