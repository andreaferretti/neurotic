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

import neo
import ./core

proc classify*[A: SomeReal](m: Layer[A], input: Vector[A]): int =
  let output = m.forward(input)
  var maxValue = output[0]
  for i, v in output:
    if v >= maxValue:
      maxValue = v
      result = i

proc evaluate*[A: SomeReal](m: Layer[A], testData: seq[(Vector[A], int)]): int =
  result = 0
  for x in testData:
    let (input, label) = x
    if label == m.classify(input):
      result += 1