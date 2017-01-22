import neurotic, linalg

when isMainModule:
  let
    l1 = dense(784, 50)
    l2 = dense(50, 10)
    cost = QuadraticCost()
  var
    m1 = l1.withMemory
    m2 = sigmoidModule()
    # m2 = reluModule()
    m3 = l2.withMemory
    # m4 = m1 -> m2 -> m3
    m4 = sequential(m1, m2, m3).asModule64
  # let
  #   v = randomVector(784).toDynamic
  #   w = randomVector(20).toDynamic

  let data = mnistTrainData()
  for d in data:
    let (input, output) = d
    let result = run(m4, cost, input.asVector, output)
    echo result.loss