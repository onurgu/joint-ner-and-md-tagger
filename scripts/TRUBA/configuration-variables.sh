#!/usr/bin/env bash

export ner_tagger_root=/truba/home/ogungor/projects/research/projects/focus/joint_md_and_ner/ner-tagger-dynet

#virtualenvwrapper_path=/usr/local/bin/virtualenvwrapper.sh
#virtualenv_name=dynet
export virtualenvwrapper_path=/truba/home/ogungor/.local/bin/virtualenvwrapper.sh
export virtualenv_name=joint_ner_dynet

#environment_variables_path=environment-variables
export environment_variables_path='/truba/sw/centos7.3/comp/intel/PS2017-update1/mkl/bin/mklvars.sh intel64'

export datasets_root=/truba/home/ogungor/projects/research/datasets/joint_ner_dynet/

export experiment_logs_path=/truba/home/ogungor/projects/research/projects/focus/joint_md_and_ner/experiment-logs/

#sacred_args='-m localhost:17017:joint_ner_and_md'
export sacred_args='-F '${experiment_logs_path}