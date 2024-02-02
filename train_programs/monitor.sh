#!/bin/bash

# Log file paths
pre="/exports/sascstudent/svanderwal2/programs/train_programs"
cpu_log="${pre}/logs/cpu_usage.log"
mem_log="${pre}/logs/mem_usage.log"
gpu_log="${pre}/logs/gpu_usage.log"
> $cpu_log
> $mem_log
> $gpu_log

#write some info about the header for the CPU info logs
echo "PID: porcess id" >> $cpu_log
echo "PR: process priority, the lower, the higher priority" >> $cpu_log
echo "VIRT: total virt memory" >> $cpu_log
echo "USER: user name of owner task" >> $cpu_log
echo "RES: how much phyiscal RAM is used, in kb" >> $cpu_log
echo "TIME+: CPU Time" >> $cpu_log
top -b -n 1 | egrep "PID" >> $cpu_log


while true
do
    # Log CPU usage
    top -b -n 1 | egrep "svander" >> $cpu_log
    echo "#new iteration" >> $cpu_log


    # Log GPU usage
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> $gpu_log

    # Wait for some time before the next logging (e.g., 10 seconds)
    sleep 10
done
