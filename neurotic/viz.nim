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

import linalg, nimPNG

proc savePNG*(x: DMatrix64 or DMatrix32, name: string): bool =
  let v = x.asVector
  let (M, N) = x.dim
  var s = newString(v.len)
  for i in 0 ..< v.len:
    s[i] = char(int8(v[i] * 256))
  return savePNG(name, s, LCT_GREY, 8, M, N)