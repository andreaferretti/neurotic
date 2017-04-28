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

import linalg, math

type
  QuadraticCost* = object
  CrossEntropyCost* = object

proc forward*(m: QuadraticCost, x, y: DVector32): float32 = l_2(x - y)

proc forward*(m: QuadraticCost, x, y: DMatrix32): float32 = l_2(x - y)

proc forward*(m: QuadraticCost, x, y: DVector64): float64 = l_2(x - y)

proc forward*(m: QuadraticCost, x, y: DMatrix64): float64 = l_2(x - y)

proc backward*(m: QuadraticCost, x, y: DVector32): DVector32 = 2 * (x - y)

proc backward*(m: QuadraticCost, x, y: DMatrix32): DMatrix32 = 2 * (x - y)

proc backward*(m: QuadraticCost, x, y: DVector64): DVector64 = 2 * (x - y)

proc backward*(m: QuadraticCost, x, y: DMatrix64): DMatrix64 = 2 * (x - y)

proc forward*(m: CrossEntropyCost, x, y: DVector32): float32 =
  -(x * y) # + ln(l_1(x))

proc forward*(m: CrossEntropyCost, x, y: DMatrix32): float32 =
  -(x.asVector * y.asVector) # + ln(l_1(x))

proc forward*(m: CrossEntropyCost, x, y: DVector64): float64 =
  -(x * y) # + ln(l_1(x))

proc forward*(m: CrossEntropyCost, x, y: DMatrix64): float64 =
  -(x.asVector * y.asVector) # + ln(l_1(x))

proc backward*(m: CrossEntropyCost, x, y: DVector32): DVector32 =
  -1 * y
  # let e = 1'f32 / l_1(x)
  # return constantVector(x.len, e) - y

proc backward*(m: CrossEntropyCost, x, y: DMatrix32): DMatrix32 =
  -1 * y
  # let e = 1'f32 / l_1(x)
  # let (m, n) = x.dim
  # return constantMatrix(m, n, e) - y

proc backward*(m: CrossEntropyCost, x, y: DVector64): DVector64 =
  -1 * y
  # let e = 1'f64 / l_1(x)
  # return constantVector(x.len, e) - y

proc backward*(m: CrossEntropyCost, x, y: DMatrix64): DMatrix64 =
  -1 * y
  # let e = 1'f64 / l_1(x)
  # let (m, n) = x.dim
  # return constantMatrix(m, n, e) - y