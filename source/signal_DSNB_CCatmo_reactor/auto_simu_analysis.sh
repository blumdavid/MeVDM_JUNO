#!/bin/sh
#
# auto_simu_analysis.py is a script to automate the simulation and analysis of the indirect MeV-Dark Matter search with JUNO!
#
# INPUT: the simulated spectra of the signal for different DM masses and the different backgrounds
#
# Script: the script generates datasets and analyzes the datasets corresponding to the simulated spectra FOR SEVERAL MASSES
#
# The script should be used on the IHEP cluster!
#
# Run the script in the directory "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"


# run the script "auto_gen_dataset.py". It creates all the necessary folders and generates datasets for all given Dark Matter masses:
python auto_gen_dataset.py

# To analyze the dataset for the different Dark Matter masses create one Condor description file for each mass.
# The condor files are also saved in the directory "/junofs/users/dblum/work/signal_DSNB_CCatmo_reactor"
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_25
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_35
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_45
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_55
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_65
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_75
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_85
condor_submit /junofs/users/dblum/work/signal_DSNB_CCatmo_reactor/condor_desc_file_95

# Check the status of the jobs and check, if they have finished:
python check_job_status.py

# Analyze the results of the MCMC analysis and save the output with output_analysis_v3_server.py:
python auto_output_analysis.py

# Build archive of the dataset_output_{} folders on the server:
tar -cf dataset_output_25.tar dataset_output_25/
tar -cf dataset_output_35.tar dataset_output_35/
tar -cf dataset_output_45.tar dataset_output_45/
tar -cf dataset_output_55.tar dataset_output_55/
tar -cf dataset_output_65.tar dataset_output_65/
tar -cf dataset_output_75.tar dataset_output_75/
tar -cf dataset_output_85.tar dataset_output_85/
tar -cf dataset_output_95.tar dataset_output_95/


