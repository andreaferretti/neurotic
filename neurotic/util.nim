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
import linalg

proc sumColumns*(m: DMatrix32 or DMatrix64): auto =
  let (a, _) = m.dim
  when m is DMatrix32:
    result = zeros(a, float32)
  else:
    result = zeros(a)
  for col in columns(m):
    result += col

proc repeat*(a: DVector32 or DVector64, n: int): auto =
  makeMatrix(a.len, n, proc(i, j: int): auto = a[i])

proc oneHot*(i, size: int): DVector64 =
  result = zeros(size)
  result[i] = 1.0

proc split*(v: DVector32 or DVector64, sizes: seq[int]): auto =
  assert v.len == foldl(sizes, a + b)
  var count = 0
  result = newSeq[type(v)]()
  for size in sizes:
    result.add(v[count ..< count + size])
    count += size