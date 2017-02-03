import linalg

# External (tensor) product
proc `.*`*(a, b: DVector64): DMatrix64 =
  makeMatrix(a.len, b.len, proc(i, j: int): float64 = a[i] * b[j])

proc sumColumns*(m: DMatrix64): DVector64 =
  let (a, _) = m.dim
  result = zeros(a)
  for col in columns(m):
    result += col

proc repeat*(a: DVector64, n: int): DMatrix64 =
  makeMatrix(a.len, n, proc(i, j: int): float64 = a[i])

proc batch*(xs: varargs[DVector64]): DMatrix64 = matrix(@xs).t