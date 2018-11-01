#!/usr/bin/env bash

INPUT=$1
OUTPUT=$2

cat $INPUT | ./evaluation/conlleval > $OUTPUT
