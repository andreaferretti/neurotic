import streams, endians, sequtils
import linalg

proc readInt32BE(s: FileStream): int =
  var r = s.readInt32()
  var tmp: int32
  bigEndian32(addr tmp, addr r)
  result = tmp

proc load_img_file(img_file: string, max_entries=high(int)): seq[DMatrix64] =
  var s = newFileStream(img_file, fmRead)
  let magic = s.readInt32BE()
  assert(magic == 2051)
  let numImages = s.readInt32BE()
  let rows = s.readInt32BE()
  let columns = s.readInt32BE()

  result = @[]
  for n in 0..<min(numImages, max_entries):
    var img = zeros(rows, columns)
    for i in 0..<rows:
      for j in 0..<columns:
        img[i, j] = toFloat(cast[int](s.readChar())) / 255.0
    result.add(img)
  s.close()

proc load_label_file(label_file: string, max_entries=high(int)): seq[int] =
  var s = newFileStream(label_file, fmRead)
  let magic = s.readInt32BE()
  assert(magic == 2049)
  let numItems = s.readInt32BE()

  result = @[]
  for n in 0..<min(numItems, max_entries):
    result.add(cast[int](s.readChar()))
  s.close()

proc mnist_load*(img_file, label_file: string, max_entries=high(int)): auto =
  proc vectorize(i: int): DVector64 =
    let m = 10
    result = zeros(m)
    result[i] = 1.0

  let images = load_img_file(img_file, max_entries=max_entries)
  let labels = load_label_file(label_file, max_entries=max_entries)
  assert(len(images) == len(labels))
  result = zip(images, labels.map(vectorize))