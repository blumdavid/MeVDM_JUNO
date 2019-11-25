#!/bin/sh
#
# auto_simu_analysis_S90_v2.py is a script to automate the simulation and analysis of the indirect MeV-Dark Matter search with JUNO!
#
# In the case of NO signal contribution in the simulated spectra -> S90_DSNB_CCatmo_reactor_NCatmo_FN
#
# INPUT: the simulated spectra of the signal for different DM masses and the different backgrounds
#
# Script: analyzes the datasets corresponding to the simulated spectra FOR SEVERAL MASSES and saves the output of the analysis!
#
# The script should be used on the IHEP cluster!
#
# Run the script in the directory "/junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN"

# The datasets are the same for all DM masses and saved in the folder "S90_DSNB_CCatmo_reactor_NCatmo_FN/dataset_output_0/datasets"

# To analyze the dataset for the different Dark Matter masses create one Condor description file for each mass.
# The condor files are also saved in the directory "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor_NCatmo_FN"
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_15
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_20
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_25
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_30
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_35
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_40
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_45
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_50
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_55
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_60
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_65
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_70
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_75
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_80
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_85
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_90
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_95
condor_submit /junofs/users/dblum/work/S90_DSNB_CCatmo_reactor_NCatmo_FN/condor_desc_file_100

# Check the status of the jobs and check, if they have finished:
python check_job_status_v2.py

# Analyze the results of the MCMC analysis and save the output with output_analysis_v4_server.py and output_analysis_v4.py:
python auto_output_analysis_v2.py

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


