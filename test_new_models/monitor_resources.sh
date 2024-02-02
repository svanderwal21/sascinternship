#!/bin/bash

LOG_FILE="/exports/sascstudent/svanderwal2/programs/test_new_models/logs_from_slurm/log_${SLURM_JOB_ID}.log"

# Function to log system resources
log_resources() {
    echo "Logging system resources to $LOG_FILE..."
    while true; do
        # Timestamp
        echo "Timestamp: $(date)" >> $LOG_FILE

        # Log GPU usage using nvidia-smi
        echo "GPU Usage:" >> $LOG_FILE
        nvidia-smi >> $LOG_FILE

        # Log CPU and Memory usage using free and top
        echo "CPU and Memory Usage:" >> $LOG_FILE
        top -bn1 | head -20 >> $LOG_FILE
        echo "Memory Usage:" >> $LOG_FILE
        free -h >> $LOG_FILE

        # Wait for 60 seconds before next log
        sleep 60
    done
}

# Call the function to start logging
log_resources

