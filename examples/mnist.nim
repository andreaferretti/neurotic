# Copyright 2017 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    l1 = dense(784, 512)
    l2 = dense(512, 10)
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