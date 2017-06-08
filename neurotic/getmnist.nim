# Copyright 2017 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streams, endians, sequtils, os, httpclient
import neo

proc readInt32BE(s: FileStream): int =
  var r = s.readInt32()
  var tmp: int32
  bigEndian32(addr tmp, addr r)
  result = tmp

proc loadImgFile(imgFile: string, maxEntries = high(int)): seq[Matrix[float64]] =
  var s = newFileStream(imgFile, fmRead)
  let magic = s.readInt32BE()
  assert(magic == 2051)
  let numImages = s.readInt32BE()
  let rows = s.readInt32BE()
  let columns = s.readInt32BE()

  result = @[]
  for n in 0 ..< min(numImages, maxEntries):
    var img = zeros(rows, columns)
    for i in 0..<columns:
      for j in 0..<rows:
        img[j, i] = toFloat(cast[int](s.readChar())) / 255.0
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
  let images = loadImgFile(imgFile, maxEntries = maxEntries)
  let labels = loadLabelFile(labelFile, maxEntries = maxEntries)
  assert(len(images) == len(labels))
  result = zip(images, labels)

proc mnistTrainData*(dataDir = "data", maxEntries = high(int)): auto =
  mnistDownload(dataDir)
  mnistLoad(dataDir & "/train-images.idx3-ubyte", dataDir & "/train-labels.idx1-ubyte")

proc mnistTestData*(dataDir = "data", maxEntries = high(int)): auto =
  mnistDownload(dataDir)
  mnistLoad(dataDir & "/t10k-images.idx3-ubyte", dataDir & "/t10k-labels.idx1-ubyte")