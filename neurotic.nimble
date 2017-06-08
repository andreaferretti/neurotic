# Package

version       = "0.1.0"
author        = "Andrea Ferretti"
description   = "Neural networks"
license       = "Apache2"
skipDirs      = @["examples"]

# Dependencies

requires "nim >= 0.16.0", "random >= 0.5.3", "neo >= 0.1.0",
  "alea >= 0.1.1", "nimPNG >= 0.1.7"

task mnist, "compile MNIST example":
  --define:openblas
  --define:release
  --path:"."
  --run
  setCommand "c", "examples/mnist.nim"