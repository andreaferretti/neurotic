import linalg

# External (tensor) product
proc `.*`*(a, b: DVector64): DMatrix64 =
  makeMatrix(a.len, b.len, proc(i, j: int): float64 = a[i] * b[j])

# Hadamard (component-wise) product
proc `|*|`*(a, b: DVector64): DVector64 =
  assert a.len == b.len
  result = zeros(a.len)
  for i in 0 .. < a.len:
    result[i] = a[i] * b[i]

proc `|*|`*(a, b: DMatrix64): DMatrix64 =
  assert a.dim == b.dim
  let (m, n) = a.dim
  result = zeros(m, n)
  for i in 0 .. < m:
    for j in 0 .. < n:
      result[i, j] = a[i, j] * b[i, j]

proc sumColumns*(m: DMatrix64): DVector64 =
  let (a, _) = m.dim
  result = zeros(a)
  for col in columns(m):
    result += col

proc repeat*(a: DVector64, n: int): DMatrix64 =
  makeMatrix(a.len, n, proc(i, j: int): float64 = a[i])

proc batch*(xs: varargs[DVector64]): DMatrix64 = matrix(@xs).t