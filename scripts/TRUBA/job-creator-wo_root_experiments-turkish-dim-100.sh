#!/usr/bin/env bash

echo $0
rundir_path=`dirname $0`
experiment_name=${1:-TRUBA-all-experiments-20180311-01}
dim=${2:-10}

partition_name=${3:-short}
core_per_job=${4:-4}
max_time=${5:-4-00:00:00}

debug=${6:-0}
target_languages=${7:-czech spanish finnish hungarian turkish}
n_trials=${8}
extra_arguments_to_be_added_to_every_job_line=$9
extra_arguments_to_be_added_to_every_job_line="$extra_arguments_to_be_added_to_every_job_line ;"

configuration_variables_script_path=${10:-./scripts/TRUBA/configuration-variables.sh}

sub_job_id=0
max_jobs_to_submit=1000

# jobs_line_by_line=`${rundir_path}/helper-script-to-run-the-experiment-set-TRUBA.sh ${experiment_name} ${dim}`

#echo $jobs_line_by_line | while read line; do

bash ${rundir_path}/helper-script-to-run-the-wo_root-experiment-set-over-all-languages-TRUBA-turkish.sh ${experiment_name} ${dim} ${configuration_variables_script_path} ${debug} "${target_languages}" ${n_trials} | while read line; do

	sub_job_id=$((sub_job_id + 1))
	echo $sub_job_id
	echo $max_jobs_to_submit
	echo $line $extra_arguments_to_be_added_to_every_job_line

	# experiment_name=XXX-dim-10-morpho_tag_type-char
	job_id=`echo ${line} $extra_arguments_to_be_added_to_every_job_line | awk '{ match($0, /.* experiment_name=([^ ]+) /, arr); printf "%s", arr[1]; }'`

    n_current_jobs=$(squeue -o '%i' -h | wc -l)
    while [[ n_current_jobs -ge 100 ]]; do
        echo Unfortunately, the number of jobs waiting and running ${n_current_jobs} is equal to or greater than the limit of 100. Waiting for 5 minutes to check again.
        sleep 300
        n_current_jobs=$(squeue -o '%i' -h | wc -l)
    done

	echo '#!/bin/bash' > ${rundir_path}/batch-script-${job_id}.sh
	echo $line $extra_arguments_to_be_added_to_every_job_line >> ${rundir_path}/batch-script-${job_id}.sh

	RES=$(sbatch -A ogungor -J ${job_id} -p ${partition_name} -c ${core_per_job} --time=${max_time} --mail-type=END --mail-user=onurgu@boun.edu.tr ${rundir_path}/batch-script-${job_id}.sh)
	SLURM_JOB_ID=${RES##* }

	echo SLURM_JOB_ID ${SLURM_JOB_ID} sleeping for 120 seconds to allow time to FileStorageObserver
	echo ${SLURM_JOB_ID} ${sub_job_id} >> ${rundir_path}/slurm_job_ids.${job_id}.txt
	echo ${line} $extra_arguments_to_be_added_to_every_job_line >> ${rundir_path}/slurm_job_ids.${job_id}.txt
	sleep 120

	if [[ sub_job_id -eq max_jobs_to_submit ]]; then
		# echo exit
		exit
	fi
done