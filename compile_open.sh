#! /bin/bash

focus=""
output="mlcfgs/proof.mlcfg"
if [ -z ${1+x} ]
then
  focus=""
else
  focus="--focus-on=$1"
  output="mlcfgs/$1.mlcfg"
fi


CARGO_TARGET_DIR=creusot cargo clean -p interpreter &&
CARGO_TARGET_DIR=creusot cargo creusot $focus --span-mode=absolute --output-file=$output -- -p interpreter &&
#  ~/creusot-experiments/creusot/ide $output
 ./ide $output
