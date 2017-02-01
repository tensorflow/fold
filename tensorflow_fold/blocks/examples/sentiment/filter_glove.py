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
"""Utility to filter GloVe vectors by vocabulary.

Vectors: <http://nlp.stanford.edu/data/glove.840B.300d.zip>
Sentences: <http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip>
(file is SOStr.txt).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
# import google3
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('glove_file', None, 'GloVe file.')
flags.DEFINE_string('sentence_file', None, 'Sentence file (one per line).')
flags.DEFINE_string('word_separator', '|', 'Word separator.')
flags.DEFINE_string('output_file', None, 'Output file')


def main(_):
  vocab = set()
  with codecs.open(FLAGS.sentence_file, encoding='utf-8') as f:
    for line in f:
      # Drop the trailing newline and strip backslashes. Split into words.
      vocab.update(line.strip().replace('\\', '').split(FLAGS.word_separator))

  nread = 0
  nwrote = 0
  with codecs.open(FLAGS.glove_file, encoding='utf-8') as f:
    with codecs.open(FLAGS.output_file, 'w', encoding='utf-8') as out:
      for line in f:
        nread += 1
        line = line.strip()
        if not line: continue
        if line.split(u' ', 1)[0] in vocab:
          out.write(line + '\n')
          nwrote += 1

  print('read %s lines, wrote %s' % (nread, nwrote))

if __name__ == '__main__':
  tf.app.run()
