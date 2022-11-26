#!/bin/bash
conda activate jackhrenviron
input_folder=input_folder/2017_script

# Low mem
python program.py $input_folder/script_input_low.txt
mem_results="low_mem"
python memory_mapping.py 2017 $mem_results
python process_data.py $mem_results

mem_results="low_mem_sparse"
python memory_mapping.py 2017 $mem_results
python process_data.py $mem_results

# med mem
python program.py $input_folder/script_input_med.txt
mem_results="med_mem"
python memory_mapping.py 2017 $mem_results
python process_data.py $mem_results

mem_results="med_mem_sparse"
python memory_mapping.py 2017 $mem_results
python process_data.py $mem_results

# high mem
python program.py $input_folder/script_input_high.txt
mem_results="high_mem"
python memory_mapping.py 2017 $mem_results
python process_data.py $mem_results

mem_results="high_mem_sparse"
python memory_mapping.py 2017 $mem_results
python process_data.py $mem_results