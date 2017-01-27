import linalg
import ./core

proc classify*(m: Module64, input: DVector64): int =
  let output = m.forward(input)
  var maxValue = output[0]
  for i, v in output:
    if v >= maxValue:
      maxValue = v
      result = i

proc evaluate*(m: Module64, testData: seq[(DVector64, int)]): int =
  result = 0
  for x in testData:
    let (input, label) = x
    if label == m.classify(input):
      result += 1