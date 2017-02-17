import linalg

type QuadraticCost* = object

proc forward*(m: QuadraticCost, x, y: DVector32): float32 = l_2(x - y)

proc forward*(m: QuadraticCost, x, y: DMatrix32): float32 = l_2(x - y)

proc forward*(m: QuadraticCost, x, y: DVector64): float64 = l_2(x - y)

proc forward*(m: QuadraticCost, x, y: DMatrix64): float64 = l_2(x - y)

proc backward*(m: QuadraticCost, x, y: DVector32): DVector32 = 2 * (x - y)

proc backward*(m: QuadraticCost, x, y: DMatrix32): DMatrix32 = 2 * (x - y)

proc backward*(m: QuadraticCost, x, y: DVector64): DVector64 = 2 * (x - y)

proc backward*(m: QuadraticCost, x, y: DMatrix64): DMatrix64 = 2 * (x - y)