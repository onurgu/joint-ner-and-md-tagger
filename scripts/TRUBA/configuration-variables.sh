#!/usr/bin/env bash
export ner_tagger_root=/truba/home/ogungor/projects/research/projects/focus/joint_md_and_ner/joint-ner-and-md-tagger

export virtualenv_name=joint_ner_dynet

export datasets_root=~/projects/research/datasets/joint_ner_dynet/

export experiment_logs_path=/truba/home/ogungor/projects/research/projects/focus/joint_md_and_ner/experiment-logs/
export virtualenvwrapper_path=/truba/home/ogungor/.local/bin/virtualenvwrapper.sh
export environment_variables_path=environment-variables
export sacred_args='-F '${experiment_logs_path}

export pretrained_embeddings=turkish/huawei-turkish-texts-dim-10.vec
