#!/usr/bin/env bash

for label in dev test train; do
    cat ./dataset/gungor.ner.${label}.14.only_consistent | python ./scripts/converters/gungor2conllu.py > ./dataset/gungor.ner.${label}.conllu ;
done