import linalg

type
  Cost64* = concept m
    # var x: DMatrix64
    var y, t: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y, t) is float64
    # m.backward(x) is DMatrix64
    m.backward(y, t) is DVector64
  Module64* = ref object of RootObj

method forward*(m: Module64, v: DVector64): DVector64 {.base.} =
  quit "to override!"

method forward*(m: Module64, v: DMatrix64): DMatrix64 {.base.} =
  quit "to override!"

method backward*(m: Module64, v: DVector64, eta: float64): DVector64 {.base.} =
  quit "to override!"