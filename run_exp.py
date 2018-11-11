import os

command = """python src/cifar10/main.py \
  --data_format="NCHW" \
  --search_for="macro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --batch_size=128 \
  --controller_entropy_weight=0.5 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads \
  --child_num_layers=12 \
  --child_out_filters=36 \
  --child_l2_reg=0.00025 \
  --child_num_branches=6 \
  --child_num_cell_layers=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_max=0.05 \
  --child_lr_min=0.005 \
  --child_lr_T_0=60 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=20 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.5 \
  --controller_op_tanh_reduce=2.5 \
  --controller_skip_target=0.4 \
  --controller_skip_weight=0.8"""

experiments = {"new_1_1": 50, "new_1_2": 50, "new_1_3": 50,
"new_2_1": 100, "new_2_2": 100, "new_2_3": 100,
"new_3_1": 150, "new_3_2": 150, "new_3_3": 150}
new_command = None
for name, epochs in experiments.items():
  new_command = command + " --output_dir=\"" + name + "\""
  new_command += " --num_epochs=" + str(epochs)
  print(new_command)
  #os.system(new_command)
