import linalg

# External (tensor) product
proc `.*`*(a, b: DVector64): DMatrix64 =
  makeMatrix(a.len, b.len, proc(i, j: int): float64 = a[i] * b[j])

# Hadamard (component-wise) product
proc `|*|`*(a, b: DVector64): DVector64 =
  assert a.len == b.len
  result = newSeq[float64](a.len)
  for i in 0 .. < a.len:
    result[i] = a[i] * b[i]