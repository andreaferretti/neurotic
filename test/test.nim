import neurotic, linalg

when isMainModule:
  let
    l1 = dense(784, 30)
    l2 = dense(30, 20)
    cost = QuadraticCost()
  var
    m1 = l1.withMemory
    m2 = sigmoidModule()
    # m2 = reluModule()
    m3 = l2.withMemory
    # m4 = m1 -> m2 -> m3
    m4 = sequential(m1, m2, m3)
  let
    v = randomVector(784).toDynamic
    w = randomVector(20).toDynamic

  let result = run(m4.asModule64, cost, v, w)
  echo result.gradient
  echo result.loss