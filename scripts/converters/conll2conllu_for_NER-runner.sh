#!/usr/bin/env bash


datasets_dir=${1:-/Users/onur/Desktop/projects/research/datasets}

for lang in czech spanish finnish hungarian; do

    for label in dev test train; do

        if [ -f ${datasets_dir}/${lang}/${lang}-ner-${label}.conll ]; then

            echo ${datasets_dir}/${lang}/${lang}-ner-${label}.conll
            cat ${datasets_dir}/${lang}/${lang}-ner-${label}.conll | \
            python ./scripts/converters/conll2conllu_for_NER.py > ${datasets_dir}/${lang}/${lang}-ner-${label}.conllu

        fi

    done

done