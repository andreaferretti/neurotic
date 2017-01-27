import sequtils
import neurotic, linalg

proc adjustTest(x: tuple[a: DMatrix64, b: DVector64]): (DVector64, int) =
  let (m, v) = x
  let (i, _) = maxIndex(v)
  (m.asVector, i)

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

  let data = mnistTrainData()
  var count = 0
  # let (x, _) = data[0]
  # let (y, _) = data[1]
  # let z = batch(x.asVector, y.asVector)
  # let a1 = m1.forward(z)
  # let a2 = m1.forward(x.asVector)
  # let a3 = m1.forward(y.asVector)
  # echo a1 =~ batch(a2, a3)
  for d in data:
    let (input, output) = d
    let result = run(m4, cost, input.asVector, output)
    if count mod 1000 == 0:
      echo result.loss
    count += 1

  let testData = mnistTestData().map(adjustTest)
  let rightAnswers = m4.evaluate(testData)
  let perc = rightAnswers.float * 100.0 / testData.len.float
  echo "Right answers: ", rightAnswers, " out of ", testData.len
  echo "Perc: ", perc, "%"

when isMainModule:
  main()