#!/bin/bash
#
# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# See https://tatoeba.org/eng/downloads for details.

# Produces a file named sentences.csv
cd /tmp/
wget http://downloads.tatoeba.org/exports/sentences.tar.bz2 -O - | tar -xjf -

# Keep only the sentences in the eight languages listed whose sentences are
# made entirely from lower case a-z letters.
awk -F"\t" '$2 ~ /(deu|eng|epo|fra|ita|nld|por|spa)/ && $3 ~ /^[\x00-\x80]+$/' < sentences.csv \
  | tr -d '[:punct:]' | tr '[:upper:]' '[:lower:]' | shuf \
> roman_sentences.csv

# Do an 20/80 dev/train split deteriministically by the last digit of the
# sentence ID number.
awk -F"\t" '$1 ~ /(1|2)\y/ {print $2","$3}' < roman_sentences.csv > roman_sentences_dev.csv
awk -F"\t" '$1 !~ /(1|2)\y/ {print $2","$3}' < roman_sentences.csv > roman_sentences_train.csv
