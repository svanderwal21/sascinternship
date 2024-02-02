python run_clm.py \
    --model_name_or_path /exports/sascstudent/svanderwal2/programs/BioGPT-Large \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
