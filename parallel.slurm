#!/bin/bash
#SBATCH -J parallel
#SBATCH --qos=normal
#SBATCH --partition cpu
#SBATCH --nodelist=cu0[1-4]
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00

WORKDIR=`pwd`
parallel --workdir $WORKDIR --joblog log --max-proc 50% --sshloginfile computers < tasks
