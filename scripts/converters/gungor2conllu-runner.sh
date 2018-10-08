#!/usr/bin/env bash

datasets_dir=${1:-/Users/onur/Desktop/projects/research/datasets}

for label in dev test train; do
    cat ${datasets_dir}/turkish/gungor.ner.${label}.14.only_consistent | python ./scripts/converters/gungor2conllu.py > ${datasets_dir}/turkish/turkish-ner-${label}.conllu ;
done

# Turkish
# cat ~/Desktop/projects/research/datasets/turkish/train.merge.utf8.gungor_format | python ./scripts/converters/gungor2conllu.py > ~/Desktop/projects/research/datasets/turkish/train.merge.utf8.conllu
# cat ~/Desktop/projects/research/datasets/turkish/test.merge.utf8.gungor_format | python ./scripts/converters/gungor2conllu.py > ~/Desktop/projects/research/datasets/turkish/test.merge.utf8.conllu

 cat ~/Desktop/projects/research/datasets/turkish/train.merge.utf8.gungor_format | python ./scripts/converters/gungor2conllu.py > ~/Desktop/projects/research/datasets/turkish/turkish-md-train.conllu
 cat ~/Desktop/projects/research/datasets/turkish/test.merge.utf8.gungor_format | python ./scripts/converters/gungor2conllu.py > ~/Desktop/projects/research/datasets/turkish/turkish-md-test.conllu