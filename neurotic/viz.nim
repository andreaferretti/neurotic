import linalg, nimPNG

proc savePNG*(x: DMatrix64 or DMatrix32, name: string): bool =
  let v = x.asVector
  let (M, N) = x.dim
  var s = newString(v.len)
  for i in 0 ..< v.len:
    s[i] = char(int8(v[i] * 256))
  return savePNG(name, s, LCT_GREY, 8, M, N)