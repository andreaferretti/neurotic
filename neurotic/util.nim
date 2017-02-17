import linalg

proc sumColumns*(m: DMatrix32 or DMatrix64): auto =
  let (a, _) = m.dim
  when m is DMatrix32:
    result = zeros(a, float32)
  else:
    result = zeros(a)
  for col in columns(m):
    result += col

proc repeat*(a: DVector32 or DVector64, n: int): auto =
  makeMatrix(a.len, n, proc(i, j: int): auto = a[i])