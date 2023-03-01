#!/bin/bash
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir data
gsutil -m cp gs://omnimatte/data/kubric-shadows-v1-train.tar.gz data/
gsutil -m cp gs://omnimatte/data/kubric-shadows-v1-test.tar.gz data/
gsutil -m cp gs://omnimatte/data/kubric-shadows-v2-train.tar.gz data/
gsutil -m cp gs://omnimatte/data/kubric-reflections-v1-train.tar.gz data/
gsutil -m cp gs://omnimatte/data/kubric-reflections-v1-test.tar.gz data/
gsutil -m cp gs://omnimatte/data/kubric-reflections-v2-train.tar.gz data/
gsutil -m cp gs://omnimatte/data/real.tar.gz data/
cd data
for x in *tar.gz; do
  echo "extracting $x..."
  tar -xf $x
done
cd ..
