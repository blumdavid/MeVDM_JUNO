#!/bin/bash

# job_100.sh:
#
# file to submit jobs for S90_DSNB_CCatmo_reactor_NCatmo_FN:
#
# use to analyze the NO signal case.
# corresponding to DM mass = 100 MeV
#
# Info about job submission: http://afsapply.ihep.ac.cn/cchelp/en/local-cluster/jobs/HTCondor/#3213-tips-of-using-hepjob

python /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/analyze_spectra_v8_server2.py $1 100
