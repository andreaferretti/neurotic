# Package

version       = "0.1.0"
author        = "Andrea Ferretti"
description   = "Neural networks"
license       = "Apache2"
skipDirs      = @["examples"]

# Dependencies

requires "nim >= 0.16.0", "random >= 0.5.3", "linalg >= 0.6.6",
  "alea >= 0.1.1", "nimPNG >= 0.2.0"

task mnist, "compile MNIST example":
  --define:openblas
  --define:release
  --path:"."
  --run
  setCommand "c", "examples/mnist.nim"

task mnist32, "compile MNIST example (32-bit)":
  --define:openblas
  --define:release
  --path:"."
  --run
  setCommand "c", "examples/mnist32.nim"