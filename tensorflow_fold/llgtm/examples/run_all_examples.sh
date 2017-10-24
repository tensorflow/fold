# To run, type "source examples/run_all_examples.sh"

bazel run -c opt examples:character_rnn -- --num_steps=100 --alsologtostderr
bazel run -c opt examples:tree_rnn -- --num_steps=100 --alsologtostderr
