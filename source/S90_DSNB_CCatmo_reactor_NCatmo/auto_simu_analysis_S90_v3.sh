#!/bin/sh
#
# auto_simu_analysis_S90_v3.py is a script to automate the simulation and analysis of the indirect MeV-Dark Matter search with JUNO!
#
# In the case of NO signal contribution in the simulated spectra -> S90_DSNB_CCatmo_reactor_NCatmo
#
# INPUT: the simulated spectra of the signal for different DM masses and the different backgrounds
#
# Script: analyzes the datasets corresponding to the simulated spectra FOR SEVERAL MASSES and saves the output of the analysis!
#
# The script should be used on the IHEP cluster!
#
# Run the script in the directory "/junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo"

# The datasets are the same for all DM masses and saved in the folder "S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_0/datasets"

# To analyze the dataset for the different Dark Matter masses create one job submission file for each mass.
# The job sub-files are also saved in the directory "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor_NCatmo"
hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_15/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_15/err%{ProcId} job_15.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_20/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_20/err%{ProcId} job_20.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_25/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_25/err%{ProcId} job_25.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_30/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_30/err%{ProcId} job_30.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_35/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_35/err%{ProcId} job_35.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_40/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_40/err%{ProcId} job_40.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_45/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_45/err%{ProcId} job_45.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_50/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_50/err%{ProcId} job_50.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_55/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_55/err%{ProcId} job_55.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_60/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_60/err%{ProcId} job_60.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_65/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_65/err%{ProcId} job_65.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_70/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_70/err%{ProcId} job_70.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_75/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_75/err%{ProcId} job_75.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_80/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_80/err%{ProcId} job_80.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_85/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_85/err%{ProcId} job_85.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_90/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_90/err%{ProcId} job_90.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_95/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_95/err%{ProcId} job_95.sh

hep_sub -g juno -u vanilla -os SL6 -argu %{ProcId} -n 100 -o /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_100/out%{ProcId} -e /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo/dataset_output_100/err%{ProcId} job_100.sh

# Check the status of the jobs and check, if they have finished:
python check_job_status_v3.py

# Analyze the results of the MCMC analysis and save the output with output_analysis_v5_server.py and output_analysis_v5.py:
python auto_output_analysis_v3.py

# Build archive of the dataset_output_{} folders on the server:
tar -cf dataset_output_15.tar dataset_output_15/
tar -cf dataset_output_20.tar dataset_output_20/
tar -cf dataset_output_25.tar dataset_output_25/
tar -cf dataset_output_30.tar dataset_output_30/
tar -cf dataset_output_35.tar dataset_output_35/
tar -cf dataset_output_40.tar dataset_output_40/
tar -cf dataset_output_45.tar dataset_output_45/
tar -cf dataset_output_50.tar dataset_output_50/
tar -cf dataset_output_55.tar dataset_output_55/
tar -cf dataset_output_60.tar dataset_output_60/
tar -cf dataset_output_65.tar dataset_output_65/
tar -cf dataset_output_70.tar dataset_output_70/
tar -cf dataset_output_75.tar dataset_output_75/
tar -cf dataset_output_80.tar dataset_output_80/
tar -cf dataset_output_85.tar dataset_output_85/
tar -cf dataset_output_90.tar dataset_output_90/
tar -cf dataset_output_95.tar dataset_output_95/
tar -cf dataset_output_100.tar dataset_output_100/
