# Package

version       = "0.1.0"
author        = "Andrea Ferretti"
description   = "Neural networks"
license       = "Apache2"

# Dependencies

requires "nim >= 0.16.0", "random >= 0.5.3", "linalg >= 0.5.3",
  "alea >= 0.1.1"

task run, "run example":
  --define:openblas
  --define:release
  # --run
  --path:"."
  setCommand "c", "test/test.nim"