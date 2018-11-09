#!/usr/bin/env bash

experiment_name=${1:-section1-all-20180311-01}
original_experiment_name=${experiment_name}

configuration_file_path=${3:-./scripts/TRUBA/configuration-variables.sh}
debug=${4:-0}
target_languages=${5:-czech spanish finnish hungarian turkish}

if [ -f ${configuration_file_path} ]; then
    source ${configuration_file_path};
fi

file_format=conllu

preamble="cd ${ner_tagger_root} && \
          source ${virtualenvwrapper_path} && \
          workon ${virtualenv_name} && \
          source ${environment_variables_path} && \
          python control_experiments.py ${sacred_args} with debug=${debug} "

n_trials=10

dim=${2:-10}
morpho_tag_type=char

for trial in `seq 1 ${n_trials}`; do

    for batch_size in 1; do

        for lang_name in ${target_languages}; do

            lang_dataset_root=${datasets_root}/${lang_name}

            ini_filepath=${lang_dataset_root}/${lang_name}-joint-md-and-ner-tagger.ini
            lang_dataset_filepaths=`python ./utils/ini_parse.py --add_suffixes --input ${ini_filepath} --query ner.train_file ner.dev_file ner.test_file md.train_file md.dev_file md.test_file`

            # lang_dataset_root=${lang_dataset_root}
            dataset_filepaths="file_format=${file_format} lang_name=${lang_name} datasets_root=${datasets_root} ${lang_dataset_filepaths} "

            for morpho_tag_type in char ; do

                small_sizes="char_dim=$dim \
                char_lstm_dim=$dim \
                morpho_tag_dim=$dim \
                morpho_tag_lstm_dim=$dim \
                morpho_tag_type=${morpho_tag_type} \
                word_dim=$dim \
                word_lstm_dim=$dim \
                batch_size=$batch_size \
                lr_method=sgd-learning_rate_float@0.01 "
                # changed the learning rate to 0.01 from 0.100

                # experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-trial-`printf "%02d" ${trial}`
                experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}

                pre_command="echo ${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-lang_name-${lang_name}-trial-`printf "%02d" ${trial}` >> ${experiment_name}.log"

                for imode in 0 1 2 ; do
                    if [[ $imode == 0 ]]; then
                        for amodels in 1 0 ; do
                            command=${pre_command}" && "" ${preamble} \
                            active_models=${amodels} \
                            integration_mode=$imode \
                            dynet_gpu=0 \
                            embeddings_filepath=\""${pretrained_embeddings}"\" \
                            ${dataset_filepaths} \
                            $small_sizes \
                            experiment_name=${experiment_name} ;"
                            echo $command;
                        done;
                    elif [[ $imode == 1 ]]; then
                        command=${pre_command}" && "" ${preamble} \
                        active_models=2 \
                        integration_mode=1 \
                        dynet_gpu=0 \
                        embeddings_filepath=\""${pretrained_embeddings}"\" \
                        ${dataset_filepaths} \
                        $small_sizes \
                        experiment_name=${experiment_name} ;"
                        echo $command;
                    else
                        command=${pre_command}" && "" ${preamble} \
                        active_models=2 \
                        integration_mode=2 \
                        multilayer=1 \
                        shortcut_connections=1 \
                        dynet_gpu=0 \
                        embeddings_filepath=\""${pretrained_embeddings}"\" \
                        ${dataset_filepaths} \
                        $small_sizes \
                        experiment_name=${experiment_name} ;"
                        echo $command;

                        command=${pre_command}" && "" ${preamble} \
                        active_models=2 \
                        integration_mode=2 \
                        dynet_gpu=0 \
                        embeddings_filepath=\""${pretrained_embeddings}"\" \
                        ${dataset_filepaths} \
                        $small_sizes \
                        experiment_name=${experiment_name} ;"
                        echo $command;

                    fi ;
                done

            done

        done
	done
done