#!/bin/bash

# toy examples using `reddit_threads` dataset

# dataset config
dataset_name="reddit_threads"
dataset_source="reddit_threads"

# model config
model_name="tiny"
hidden_size=128
num_hidden_layers=2
echo "In Tiny setting!"

# train config
batch_size=128
workerCount=1
num_cpus=4
total_tokens=1e8
warmup_tokens=1e7
max_position_embeddings=256
tie_word_embeddings=0
lr=3e-4
weight_decay=0.01
max_grad_norm=1
eps=1e-8
pack_tokens=1
memory=40960
## deep-speed config
deepspeed_config="./examples/ds_config2_pt.json"
## tokenization config
attr_assignment="random"


# dataset config
samples_per_saving=1000000
tokenization_config="./examples/toy_examples/reddit_tokenization_config.json"
model_config="./examples/toy_examples/reddit_model_config.json"
ds_prefix="reddit"



# env config
data_dir_prefix="./data"
output_dir_prefix="./exp/models"

let batch_actual=batch_size*workerCount
data_dir="TUDataset"
output_dir="${ds_prefix}/pt_h${hidden_size}_l${num_hidden_layers}_b${batch_actual}_mpe${max_position_embeddings}_tk${total_tokens}"

if [ "$pretrain_cpt" != "" ]
then
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${pretrain_cpt}"
fi


raw_udf="
  --data_dir='${data_dir_prefix}/${data_dir}'
  --output_dir='${output_dir_prefix}/${output_dir}'
  --pretrain_cpt='${pretrain_cpt}'
  --dataset_name='${dataset_source}'
  --tokenization_config='${tokenization_config}'
  --attr_assignment='${attr_assignment}'
  --batch_size=${batch_size}
  --pack_tokens=${pack_tokens}
  --num_workers=${num_cpus}
  --max_position_embeddings=${max_position_embeddings}
  --tie_word_embeddings=${tie_word_embeddings}
  --lr=${lr}
  --weight_decay=${weight_decay}
  --eps=${eps}
  --max_grad_norm=${max_grad_norm}
  --total_tokens=${total_tokens}
  --warmup_tokens=${warmup_tokens}
  --model_config='${model_config}'
  --num_hidden_layers=${num_hidden_layers}
  --hidden_size=${hidden_size}
  --samples_per_saving=${samples_per_saving}
  --deepspeed_config='${deepspeed_config}'
  --optimization_config='${optimization_config}'
"

udf=${raw_udf//$'\n'/}

echo ${udf}

deepspeed ./examples/train_pretrain.py ${raw_udf}

echo $raw_udf
echo "Train and evaluation finished"
