import linalg

type
  Cost32* = concept c
    var x, s: DMatrix32
    var y, t: DVector32
    c.forward(x, s) is float32
    c.forward(y, t) is float32
    c.backward(x, s) is DMatrix32
    c.backward(y, t) is DVector32
  Cost64* = concept c
    var x, s: DMatrix64
    var y, t: DVector64
    c.forward(x, s) is float64
    c.forward(y, t) is float64
    c.backward(x, s) is DMatrix64
    c.backward(y, t) is DVector64
  Layer32* = ref object of RootObj
  Layer64* = ref object of RootObj

method forward*(m: Layer32, v: DVector32): DVector32 {.base.} =
  quit "to override!"

method forward*(m: Layer32, v: DMatrix32): DMatrix32 {.base.} =
  quit "to override!"

method backward*(m: Layer32, v: DVector32, eta: float32): DVector32 {.base.} =
  quit "to override!"

method backward*(m: Layer32, v: DMatrix32, eta: float32): DMatrix32 {.base.} =
  quit "to override!"

method forward*(m: Layer64, v: DVector64): DVector64 {.base.} =
  quit "to override!"

method forward*(m: Layer64, v: DMatrix64): DMatrix64 {.base.} =
  quit "to override!"

method backward*(m: Layer64, v: DVector64, eta: float64): DVector64 {.base.} =
  quit "to override!"

method backward*(m: Layer64, v: DMatrix64, eta: float64): DMatrix64 {.base.} =
  quit "to override!"