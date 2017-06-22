#! /bin/bash
echo "SLURM_JOB_ID " $SLURM_JOB_ID  "; SLURM_JOB_NAME " $SLURM_JOB_NAME "; SLURM_JOB_NODELIST " $SLURM_JOB_NODELIST "; SLURMD_NODENAME " $SLURMD_NODENAME  "; SLURM_JOB_NUM_NODES " $SLURM_JOB_NUM_NODES
source /net/people/plgtgrel/simple_venv/bin/activate
python main.py
echo "DONE"
