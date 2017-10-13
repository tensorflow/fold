#!/bin/bash
# Smoke test that runs all examples. Command-line arguments are
# forwarded to 'bazel build', e.g.
# ./tensorflow_fold/run_all_examples.sh --config=opt
set -o verbose
set -e
bazel build "$@" tensorflow_fold/...

# loom benchmark
./bazel-bin/tensorflow_fold/loom/benchmarks/iclr_2017_benchmark \
  --vector_size=8 --tree_size=4 --num_repeats=1 --alsologtostderr

## loom calculator
TMP=$(mktemp -d)
./bazel-bin/tensorflow_fold/loom/calculator_example/make_dataset \
  --output_path="${TMP}"/examples --num_samples=15 --alsologtostderr
./bazel-bin/tensorflow_fold/loom/calculator_example/train \
  --train_data_path="${TMP}"/examples --batch_size=10 --max_steps=2 \
  --logdir="${TMP}" --alsologtostderr
./bazel-bin/tensorflow_fold/loom/calculator_example/eval \
  --validation_data_path="${TMP}"/examples --eval_interval_secs=0 \
  --logdir="${TMP}" --alsologtostderr

# blocks calculator
TMP2=$(mktemp -d)
./bazel-bin/tensorflow_fold/blocks/examples/calculator/train \
  --train_data_path="${TMP}"/examples --batch_size=10 --max_steps=2 \
  --logdir="${TMP2}" --alsologtostderr

# blocks fizzbuzz
./bazel-bin/tensorflow_fold/blocks/examples/fizzbuzz/fizzbuzz \
  --validation_size=5 --steps=3 --batches_per_step=2 --alsologtostderr

# blocks mnist
TMP=$(mktemp -d)
./bazel-bin/tensorflow_fold/blocks/examples/mnist/mnist \
  --logdir_base="${TMP}" --epochs=1 --alsologtostderr

# blocks sentiment
TMP=$(mktemp -d)
echo '( 0.1 0.2' > "${TMP}"/glove
echo ') 0.3 0.4' >> "${TMP}"/glove
echo 'foo 0.1 0.2' >> "${TMP}"/glove
echo 'bar 0.3 0.4' >> "${TMP}"/glove
echo 'foo|bar|)|(|baz' > "${TMP}"/sents
echo '(3 (1 bar) (2 mu))' > "${TMP}"/train.txt
echo '(3 (1 bar) (2 mu))' > "${TMP}"/dev.txt
echo '(3 (1 bar) (2 mu))' > "${TMP}"/test.txt
./bazel-bin/tensorflow_fold/blocks/examples/sentiment/filter_glove \
  --glove_file="${TMP}"/glove --sentence_file="${TMP}"/sents \
  --output_file="${TMP}"/glove_filtered --alsologtostderr
./bazel-bin/tensorflow_fold/blocks/examples/sentiment/train \
  --checkpoint_base="${TMP}"/model --epochs=2 --tree_dir="${TMP}" \
  --embedding_file="${TMP}"/glove_filtered --alsologtostderr
./bazel-bin/tensorflow_fold/blocks/examples/sentiment/eval \
  --checkpoint_file="${TMP}"/model-1 --tree_dir="${TMP}" \
  --embedding_file="${TMP}"/glove_filtered --alsologtostderr

# blocks language_id
TMP=$(mktemp -d)
if [[ ! -f /tmp/roman_sentences.csv ]]; then
  ./tensorflow_fold/blocks/examples/language_id/fetch_datasets.sh
fi
./bazel-bin/tensorflow_fold/blocks/examples/language_id/language_id \
  --logdir_base="${TMP}" --epochs=2 --alsologtostderr
