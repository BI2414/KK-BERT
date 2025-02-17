@echo off
set PYTHONUNBUFFERED=1
python -u main.py ^
    --test 0 ^
    --name "RTE" ^
    --aug 1 ^
    --uniform 0 ^
    --gate 0 ^
    --gpu_id 1 ^
    --ratio 1.0 ^
    --read_data 0 ^
    --num_train_epochs 3 ^
    --batch_size 64 ^
    --max_len 128 ^
    --beta 0.5 ^
    --zero_peturb 0 ^
    --learning_rate 0.00005 > run_rte_aug.log 2>&1