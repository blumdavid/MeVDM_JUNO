#!/bin/bash

# job_80.sh:
#
# file to submit jobs for S90_DSNB_CCatmo_reactor_NCatmo:
#
# use to analyze the NO signal case.
# corresponding to DM mass = 80 MeV
#
# Info about job submission: http://afsapply.ihep.ac.cn/cchelp/en/local-cluster/jobs/HTCondor/#3213-tips-of-using-hepjob

python /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/analyze_spectra_v7_server2.py $1 80
