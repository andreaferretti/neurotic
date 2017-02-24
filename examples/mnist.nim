import sequtils
import neurotic, linalg, nimPNG

proc adjustTest(x: (DMatrix64, int)): (DVector64, int) =
  let (m, i) = x
  (m.asVector, i)

proc adjustTrain(x: (DMatrix64, int)): TrainingData64 =
  let (m, label) = x
  return (input: m.asVector, output: oneHot(label, 10))


proc main() =
  let
    l1 = dense(784, 50)
    l2 = dense(50, 10)
    cost = QuadraticCost()
  var
    m1 = l1.withMemory
    # m2 = sigmoidModule()
    m2 = reluModule()
    m3 = l2.withMemory
    m4 = sequential(@[m1, m2, m3])

  let data = mnistTrainData().map(adjustTrain)
  for j in 0 .. 10:
    let (x, _) = data[j]
    discard savePNG(x.asMatrix(28, 28), "mnist-" & $j & ".png")

  for _ in 1 .. 10:
    # sgd(m4, cost, data)
    miniBatchSgd(m4, cost, data)

  let testData = mnistTestData().map(adjustTest)
  let rightAnswers = m4.evaluate(testData)
  let perc = rightAnswers.float * 100.0 / testData.len.float
  echo "Right answers: ", rightAnswers, " out of ", testData.len
  echo "Perc: ", perc, "%"

when isMainModule:
  main()