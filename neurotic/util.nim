import linalg

proc sumColumns*(m: DMatrix64): DVector64 =
  let (a, _) = m.dim
  result = zeros(a)
  for col in columns(m):
    result += col

proc repeat*(a: DVector64, n: int): DMatrix64 =
  makeMatrix(a.len, n, proc(i, j: int): float64 = a[i])