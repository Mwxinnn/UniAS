python -m torch.distributed.launch --master_port=29503 --nproc_per_node=1 train_val.py -e \
 --config ./configs/VisA_config.yaml \
 --exp-path ./experiments/VisA \
