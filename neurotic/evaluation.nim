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