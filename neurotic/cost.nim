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

import neo, math
import ./util

makeUniversal(ln)

type
  QuadraticCost* = object
  CrossEntropyCost* = object

proc forward*[A: SomeReal](m: QuadraticCost, x, y: Vector[A]): A = l_2(x - y)

proc forward*[A: SomeReal](m: QuadraticCost, x, y: Matrix[A]): A = l_2(x - y)

proc backward*[A: SomeReal](m: QuadraticCost, x, y: Vector[A]): Vector[A] = 2 * (x - y)

proc backward*[A: SomeReal](m: QuadraticCost, x, y: Matrix[A]): Matrix[A] = 2 * (x - y)

proc forward*[A: SomeReal](m: CrossEntropyCost, x, y: Vector[A]): float32 =
  -(ln(x) * y) # + ln(l_1(x))

proc forward*[A: SomeReal](m: CrossEntropyCost, x, y: Matrix[A]): float32 =
  -(ln(x.asVector) * y.asVector) # + ln(l_1(x))

proc backward*[A: SomeReal](m: CrossEntropyCost, x, y: Vector[A]): Vector[A] =
  -1 * (x.inverse |*| y)

proc backward*[A: SomeReal](m: CrossEntropyCost, x, y: Matrix[A]): Matrix[A] =
  -1 * (x.inverse |*| y)