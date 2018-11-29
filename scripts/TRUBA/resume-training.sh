#!/usr/bin/env bash

experiment_name=$1
experiment_logs_dir=${2:-../experiment-logs}
maximum_epochs=${3:-100}
configuration_file_path=${4:-./scripts/TRUBA/configuration-variables.sh}
dry_run=${5:-0}

if [ -f ${configuration_file_path} ]; then
    source ${configuration_file_path};
fi

echo $0
rundir_path=`dirname $0`

partition_name=${6:-mid1}
core_per_job=${7:-8}
max_time=${8:-4-00:00:00}

sub_job_id=0
max_jobs_to_submit=1000

preamble="cd ${ner_tagger_root} && \
          source ${virtualenvwrapper_path} && \
          workon ${virtualenv_name} && \
          source ${environment_variables_path} && \
          python control_experiments.py ${sacred_args} with debug=0 "

print_resumable_experiment_configurations="python ./utils/inspect_results.py --command print_resumable_experiment_configurations --campaign_name ${experiment_name} --db_type ${experiment_logs_dir}"
echo $print_resumable_experiment_configurations

${print_resumable_experiment_configurations} | \

while read experiment_log_line; do

    sacred_experiment_args="maximum_epochs=${maximum_epochs} $experiment_log_line"

    line="$preamble $sacred_experiment_args"

    sub_job_id=$((sub_job_id + 1))
	echo $sub_job_id
	echo $max_jobs_to_submit
	echo $line

	if [[ dry_run -ne 1 ]]; then

        # experiment_name=XXX-dim-10-morpho_tag_type-char
        job_id=`echo ${line} | awk '{ match($0, /.* experiment_name=([^ ]+) /, arr); printf "%s", arr[1]; }'`

        n_current_jobs=$(squeue -o '%i' -h | wc -l)
        while [[ n_current_jobs -ge 100 ]]; do
            echo Unfortunately, the number of jobs waiting and running ${n_current_jobs} is equal to or greater than the limit of 100. Waiting for 5 minutes to check again.
            sleep 300
            n_current_jobs=$(squeue -o '%i' -h | wc -l)
        done

        echo '#!/bin/bash' > ${rundir_path}/batch-script-${job_id}.sh
        echo $line >> ${rundir_path}/batch-script-${job_id}.sh

        RES=$(sbatch -A ogungor -J ${job_id} -p ${partition_name} -c ${core_per_job} --time=${max_time} --mail-type=END --mail-user=onurgu@boun.edu.tr ${rundir_path}/batch-script-${job_id}.sh)
        SLURM_JOB_ID=${RES##* }

        echo SLURM_JOB_ID ${SLURM_JOB_ID} sleeping for 120 seconds to allow time to FileStorageObserver
        echo ${SLURM_JOB_ID} ${sub_job_id} >> ${rundir_path}/slurm_job_ids.${job_id}.txt
        echo ${line} >> ${rundir_path}/slurm_job_ids.${job_id}.txt
        sleep 120

        if [[ sub_job_id -eq max_jobs_to_submit ]]; then
            # echo exit
            exit
        fi

	fi


done