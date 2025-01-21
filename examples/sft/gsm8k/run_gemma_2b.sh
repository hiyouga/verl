set -x

torchrun --nproc-per-node=8 --master-port=28888 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_dataset=gsm8k \
    data.val_dataset=gsm8k \
    data.micro_batch_size=8 \
    model.model_path=/mnt/hdfs/veomni/models/qwen2_5-7b-instruct/ \
    trainer.save_checkpoint_path=./saves \
    trainer.experiment_name=null \
    trainer.total_epochs=2
