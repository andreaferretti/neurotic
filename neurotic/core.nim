import linalg
import collections/iface

type
  Cost64* = concept m
    # var x: DMatrix64
    var y, t: DVector64
    # m.forward(x) is DMatrix64
    m.forward(y, t) is float64
    # m.backward(x) is DMatrix64
    m.backward(y, t) is DVector64
  Module64* = distinct Interface

interfaceMethods Module64:
  forward(v: DVector64): DVector64
  forwardM(v: DMatrix64): DMatrix64
  backward(v: DVector64, eta: float64): DVector64