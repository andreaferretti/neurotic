import linalg
import ./core

template classifyT(m, input, result: untyped) =
  let output = m.forward(input)
  var maxValue = output[0]
  for i, v in output:
    if v >= maxValue:
      maxValue = v
      result = i

proc classify*(m: Layer32, input: DVector32): int = classifyT(m, input, result)

proc classify*(m: Layer64, input: DVector64): int = classifyT(m, input, result)

template evaluateT(m, testData, result: untyped) =
  result = 0
  for x in testData:
    let (input, label) = x
    if label == m.classify(input):
      result += 1

proc evaluate*(m: Layer32, testData: seq[(DVector32, int)]): int =
  evaluateT(m, testData, result)

proc evaluate*(m: Layer64, testData: seq[(DVector64, int)]): int =
  evaluateT(m, testData, result)