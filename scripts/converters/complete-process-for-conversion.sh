#!/usr/bin/env bash

datasets_dir=/Users/onur/Desktop/projects/research/datasets
target_datasets_dir=/Users/onur/Desktop/projects/research/datasets-to-TRUBA

## PART 1
## all languages except Turkish
#
## transform NER data to conllu format (without MD analyses). no transformation is required for UD based data.
#bash ./scripts/converters/conll2conllu_for_NER-runner.sh ${datasets_dir}
#
## this runs for NER and MD both
#python ./scripts/converters/conllu2conllu_with_all_analyses_runner.py ${datasets_dir}

# for NER tags
#for lang in czech finnish hungarian spanish; do
#    mkdir -p ${target_datasets_dir}/${lang}
#    cp ${datasets_dir}/${lang}/${lang}-joint-md-and-ner-tagger.ini ${target_datasets_dir}/${lang}
#    ini_filepath=${datasets_dir}/${lang}/${lang}-joint-md-and-ner-tagger.ini
#
#
#    for label in train dev test; do
#
#            lang_dataset_filepath=`python ./utils/ini_parse.py --only_values --input ${ini_filepath} --query ner.${label}_file`
#
#            ner_source_file=${lang}/${lang_dataset_filepath}.all_analyses.tagged
#            echo cp ${datasets_dir}/${ner_source_file} ${target_datasets_dir}/${ner_source_file}
#            cp ${datasets_dir}/${ner_source_file} ${target_datasets_dir}/${ner_source_file}
#
#            lang_dataset_filepath=`python ./utils/ini_parse.py --only_values --input ${ini_filepath} --query md.${label}_file`
#
#            md_source_file=${lang}/${lang_dataset_filepath}.all_analyses
#            echo cp ${datasets_dir}/${md_source_file} ${target_datasets_dir}/${md_source_file}
#            cp ${datasets_dir}/${md_source_file} ${target_datasets_dir}/${md_source_file}
#    done;
#done

# Create short versions

for file in `find ${target_datasets_dir} -name '*-ner-*.conllu.all_analyses.tagged'`; do
    echo $file; cat $file | head -100 > $file.short ; echo >> $file.short ;
done

for file in `find ${target_datasets_dir} -name '*-ud-*.conllu.all_analyses'`; do
    echo $file; cat $file | head -100 > $file.short ; echo >> $file.short ;
done

### PART 2: Turkish
#
bash ./scripts/converters/gungor2conllu-runner.sh

for lang in turkish; do
    mkdir -p ${target_datasets_dir}/${lang}
    cp ${datasets_dir}/${lang}/${lang}-joint-md-and-ner-tagger.ini ${target_datasets_dir}/${lang}
    ini_filepath=${datasets_dir}/${lang}/${lang}-joint-md-and-ner-tagger.ini

    for label in train dev test; do

            lang_dataset_filepath=`python ./utils/ini_parse.py --only_values --input ${ini_filepath} --query ner.${label}_file`

            ner_source_file=${lang}/${lang_dataset_filepath}
            echo cp ${datasets_dir}/${ner_source_file} ${target_datasets_dir}/${ner_source_file}.all_analyses.tagged
            cp ${datasets_dir}/${ner_source_file} ${target_datasets_dir}/${ner_source_file}.all_analyses.tagged

            lang_dataset_filepath=`python ./utils/ini_parse.py --only_values --input ${ini_filepath} --query md.${label}_file`

            md_source_file=${lang}/${lang_dataset_filepath}
            echo cp ${datasets_dir}/${md_source_file} ${target_datasets_dir}/${md_source_file}.all_analyses
            cp ${datasets_dir}/${md_source_file} ${target_datasets_dir}/${md_source_file}.all_analyses
    done;
done