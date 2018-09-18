#!/usr/bin/env bash


datasets_dir=/Users/onur/Desktop/projects/research/datasets

for lang in czech spanish finnish hungarian turkish; do

    for label in dev test train; do

        if [ -f ${datasets_dir}/${lang}/${lang}-ner-${label}.conll ]; then

            echo ${datasets_dir}/${lang}/${lang}-ner-${label}.conll
            cat ${datasets_dir}/${lang}/${lang}-ner-${label}.conll | \
            python ./scripts/converters/conll2conllu.py > ${datasets_dir}/${lang}/${lang}-ner-${label}.conllu

        fi

    done

done