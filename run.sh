source /home/lsl/anaconda3/envs/zeroRE/bin/activate zeroRE

for dataset in 'fewrel' 'wikizsl'
do
  for unseen in 5 10 15
  do
    for seed in 7 19 42 66 101
    do
        for k in 2 3 4
        do
        python -u main.py \
        --gpu_available 0 \
        --unseen ${unseen} \
        --k ${k} \
        --dataset ${dataset} \
        --seed ${seed} \
        --train_batch_size 32 \
        --evaluate_batch_size 640 \
        --epochs 5 \
        --lr 2e-5 \
        --warm_up 100 \
        --pretrained_model_name_or_path ../bert-base-uncased \
        --add_auto_match False
        done
    done
  done
done
