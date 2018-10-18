#!/usr/bin/env bash

TARGET_SLURM_ID=${1:-}

for file in slurm-*.out; do

    run_id=$(head -2 $file | tail -1 | sed -e 's/\"//g' -e 's/^.* //g');
    slurm_id=$(echo $file | sed 's/slurm-\([0-9]\+\)\.out/\1/');
#    echo $slurm_id $run_id;

    if [[ ${TARGET_SLURM_ID} != '' ]] ; then
     echo $slurm_id $run_id | awk '$1 == '${TARGET_SLURM_ID}' {print}' ;
    else
        echo $slurm_id $run_id;
    fi

done