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
import neo

proc sumColumns*[A: SomeReal](m: Matrix[A]): Vector[A] =
  result = zeros(m.N, A)
  for col in columns(m):
    result += col

proc repeat*[A](a: Vector[A], n: int): auto =
  makeMatrixIJ(A, a.len, n, a[i])

proc oneHot*(i, size: int): Vector[float64] =
  result = zeros(size)
  result[i] = 1.0

proc split*[A](v: Vector[A], sizes: seq[int]): auto =
  assert v.len == foldl(sizes, a + b)
  var count = 0
  result = newSeq[Vector[A]]()
  for size in sizes:
    result.add(v[count ..< count + size])
    count += size

proc inverse*[A: SomeReal](v: Vector[A]): auto =
  result = v.clone()
  for i in 0 ..< result.len:
    result[i] = 1 / result[i]

proc inverse*[A: SomeReal](m: Matrix[A]): auto =
  result = m.clone()
  for i in 0 ..< m.M * m.N:
    result.data[i] = 1 / result.data[i]

proc vertical*[A](v: Vector[A]): auto =
  v.asMatrix(v.len, 1)

proc horizontal*[A](v: Vector[A]): auto =
  v.asMatrix(1, v.len)

proc diagonal*[A: SomeReal](v: Vector[A]): Matrix[A] =
  result = zeros(v.len, v.len, A)
  for i, j in v:
    result[i, i] = v[i]