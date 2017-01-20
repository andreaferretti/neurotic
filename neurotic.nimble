# Package

version       = "0.1.0"
author        = "Andrea Ferretti"
description   = "Neural networks"
license       = "Apache2"

# Dependencies

requires "nim >= 0.16.0", "random >= 0.5.3", "linalg >= 0.5.3", "collections >= 0.3.0"

task run, "run example":
  --define:openblas
  --run
  setCommand "c", "neurotic.nim"