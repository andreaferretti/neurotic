import streams, endians, sequtils, os, httpclient
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

proc mnistDownload*(dataDir = "data") =
  discard existsOrCreateDir(dataDir)
  let
    mnistUrl = "http://yann.lecun.com/exdb/mnist/"
    files = [
      ("t10k-images", "idx3-ubyte"),
      ("t10k-labels", "idx1-ubyte"),
      ("train-images", "idx3-ubyte"),
      ("train-labels", "idx1-ubyte")
    ]
  for file in files:
    let
      (name, extension) = file
      target = dataDir / (name & "." & extension)
      path = name & "-" & extension & ".gz"
      zipped = dataDir / path
    if not existsFile(target):
      if not existsFile(zipped):
        echo "Downloading ", zipped, "..."
        downloadFile(mnistUrl & path, zipped)
      echo "Extracting ", zipped, "..."
      let outcome = execShellCmd("gzip -d -N " & zipped)
      if outcome != 0:
        raise newException(OSError, "Failed to extract " & zipped)

proc mnistLoad*(imgFile, labelFile: string, maxEntries = high(int)): auto =
  proc vectorize(i: int): DVector64 =
    let m = 10
    result = zeros(m)
    result[i] = 1.0

  let images = loadImgFile(imgFile, maxEntries = maxEntries)
  let labels = loadLabelFile(labelFile, maxEntries = maxEntries)
  assert(len(images) == len(labels))
  result = zip(images, labels.map(vectorize))

proc mnistTrainData*(dataDir = "data", maxEntries = high(int)): auto =
  mnistDownload(dataDir)
  mnistLoad(dataDir & "/train-images.idx3-ubyte", dataDir & "/train-labels.idx1-ubyte")

proc mnistTestData*(dataDir = "data", maxEntries = high(int)): auto =
  mnistDownload(dataDir)
  mnistLoad(dataDir & "/t10k-images.idx3-ubyte", dataDir & "/t10k-labels.idx1-ubyte")