#!/bin/bash

# List of job IDs
job_ids=(12464450 12464451 12464452 12464453 12464454 12464455 12464456 12464459 12464460 12464461 12464462 12464463 12464464 12464465 12464466)


# Loop over each job ID
for id in "${job_ids[@]}"
do
   # Execute the cancel_job script with the current ID
   ./cancel_job.sh $id
   # Move the corresponding output file to the discard_logs folder
   # mv "slurm-$id.out" discard_logs/
   
done