import linalg

type
  Cost64* = concept c
    var x, s: DMatrix64
    var y, t: DVector64
    c.forward(x, s) is float64
    c.forward(y, t) is float64
    c.backward(x, s) is DMatrix64
    c.backward(y, t) is DVector64
  Module64* = ref object of RootObj

method forward*(m: Module64, v: DVector64): DVector64 {.base.} =
  quit "to override!"

method forward*(m: Module64, v: DMatrix64): DMatrix64 {.base.} =
  quit "to override!"

method backward*(m: Module64, v: DVector64, eta: float64): DVector64 {.base.} =
  quit "to override!"

method backward*(m: Module64, v: DMatrix64, eta: float64): DMatrix64 {.base.} =
  quit "to override!"