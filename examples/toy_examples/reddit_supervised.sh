#!/bin/bash

#### below useless hp
dropout=0
use_mlp=0
gradient_accumulation_steps=1
memory=20480
freeze=-1
# tokenization config
ignored_off=0
attr_assignment="all"
attr_shuffle=0
# save
save_pred=1
# for eval on given ckps
eval_only=0
#### above useless hp

#====================================================================================================

# dataset config
dataset_name="reddit_threads"

# model config
model_name="tiny"
hidden_size=128
num_hidden_layers=2
echo "In Tiny setting!"

# train config
lr=1e-4
betas="[0.9,0.99]"
eps=1e-10
weight_decay=0
max_grad_norm=1

epochs=16
#epochs_warmup=$(bc <<<"scale=2;${epochs}*0.3")
epochs_warmup=4.8
batch_size_sv=256
workerCount=1
num_cpus=4
seed=42
# deep-speed config
deepspeed_config="./examples/ds_config2.json"

# supervised-task config
task_level="graph"
problem_type="single_label_classification"
num_labels=2

# dataset-specific config
samples_per_saving=0
samples_per_eval=0
k_samplers=512
ds_prefix="reddit"
# train config
max_position_embeddings_sv=256


# env config
data_dir_prefix="./data"
output_dir_prefix="./exp/models"
tokenization_config="./examples/toy_examples/reddit_tokenization_config.json"
model_config="./examples/toy_examples/reddit_model_config.json"


# pre-train ckp:: dataset-specific config
pretrain_cpt=""

let batch_size_actual=batch_size_sv*workerCount*gradient_accumulation_steps
output_folder_raw="sv_h${hidden_size}_l${num_hidden_layers}_b${batch_size_actual}_mpe${max_position_embeddings_sv}_e${epochs}"

if [ "$pretrain_cpt" = "" ]
then
  output_folder="${output_folder_raw}"
else
  output_folder="pt2${output_folder_raw}"
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${pretrain_cpt}"
fi

data_dir="TUDataset"
output_dir="${ds_prefix}/${output_folder}"


raw_udf="
  --data_dir='${data_dir_prefix}/${data_dir}'
  --output_dir='${output_dir_prefix}/${output_dir}'
  --pretrain_cpt='${pretrain_cpt}'
  --dataset_name='${dataset_name}'
  --save_pred=${save_pred}
  --tokenization_config='${tokenization_config}'
  --attr_assignment='${attr_assignment}'
  --attr_shuffle=${attr_shuffle}
  --ignored_off=${ignored_off}
  --batch_size=${batch_size_sv}
  --num_workers=${num_cpus}
  --max_position_embeddings=${max_position_embeddings_sv}
  --lr=${lr}
  --eps=${eps}
  --betas='${betas}'
  --weight_decay=${weight_decay}
  --max_grad_norm=${max_grad_norm}
  --freeze=${freeze}
  --k_samplers=${k_samplers}
  --epochs=${epochs}
  --warmup_epochs=${epochs_warmup}
  --model_config='${model_config}'
  --num_hidden_layers=${num_hidden_layers}
  --hidden_size=${hidden_size}
  --task_level=${task_level}
  --num_labels=${num_labels}
  --mlp='[${mlp}]'
  --dropout=${dropout}
  --problem_type=${problem_type}
  --samples_per_eval=${samples_per_eval}
  --deepspeed_config='${deepspeed_config}'
  --gradient_accumulation_steps=${gradient_accumulation_steps}
  --eval_only=${eval_only}
  --seed=${seed}
"

udf=${raw_udf//$'\n'/}

echo ${udf}
echo ${pretrain_cpt}

deepspeed ./examples/train_supervised.py ${raw_udf}

echo $raw_udf
echo "Train and evaluation finished"
