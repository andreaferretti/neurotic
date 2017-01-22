import streams, endians, sequtils
import linalg

proc readInt32BE(s: FileStream): int =
  var r = s.readInt32()
  var tmp: int32
  bigEndian32(addr tmp, addr r)
  result = tmp

proc loadImgFile(imgFile: string, maxEntries = high(int)): seq[DMatrix64] =
  var s = newFileStream(imgFile, fmRead)
  let magic = s.readInt32BE()
  assert(magic == 2051)
  let numImages = s.readInt32BE()
  let rows = s.readInt32BE()
  let columns = s.readInt32BE()

  result = @[]
  for n in 0 ..< min(numImages, maxEntries):
    var img = zeros(rows, columns)
    for i in 0..<rows:
      for j in 0..<columns:
        img[i, j] = toFloat(cast[int](s.readChar())) / 255.0
    result.add(img)
  s.close()

proc loadLabelFile(labelFile: string, maxEntries = high(int)): seq[int] =
  var s = newFileStream(labelFile, fmRead)
  let magic = s.readInt32BE()
  assert(magic == 2049)
  let numItems = s.readInt32BE()

  result = @[]
  for n in 0 ..< min(numItems, maxEntries):
    result.add(cast[int](s.readChar()))
  s.close()

proc mnist_load*(imgFile, labelFile: string, maxEntries = high(int)): auto =
  proc vectorize(i: int): DVector64 =
    let m = 10
    result = zeros(m)
    result[i] = 1.0

  let images = loadImgFile(imgFile, maxEntries = maxEntries)
  let labels = loadLabelFile(labelFile, maxEntries = maxEntries)
  assert(len(images) == len(labels))
  result = zip(images, labels.map(vectorize))