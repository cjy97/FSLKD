python finetune.py  \
--warmup_epoch 20 \
--max_epoch 100 \
--episodes_per_epoch 2000 \
--classifier ProtoNet \
--backbone Vit \
--init_weights "./Vit_small_mae_pretrained.pth" \
--dataset MiniImageNet \
--dataset_path "/data/fewshot/miniImageNet--ravi/" \
--way 5 \
--shot 1 \
--query 15 \
--temperature 1 \
--lr 0.0001 \
--weight_decay 0.05 \
--lr_scheduler cosine \
--augment \
--gpu 0 \
--eval_interval 1 \
# --is_prune \
# --remove_ratio 0.5 \
# --is_distill  \
# --teacher_backbone_class Res12 \
# --teacher_init_weights ../Res12-pre.pth \
# --kd_loss IRD \
# --kd_weight 0.1 \