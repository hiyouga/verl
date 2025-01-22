set -x

torchrun --nproc-per-node=8 --master-port=28888 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_dataset=openo1 \
    data.val_dataset=null \
    data.micro_batch_size=8 \
    data.total_batch_size=32 \
    data.max_seq_len=4096 \
    data.truncation=left \
    model.model_path=/mnt/hdfs/veomni/models/qwen2_5-7b-instruct/ \
    trainer.save_checkpoint_path=./saves \
    trainer.experiment_name=null \
    trainer.total_epochs=2
