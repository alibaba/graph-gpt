#!/bin/bash

# dataset config
dataset_name="ogbl-ppa"  # PCQM4Mv2 ogbg-molpcba ogbl-ppa ogbn-proteins
dataset_source=""  # molecule  PCQM4Mv2
if [ "${dataset_source}" = "" ]
then
  dataset_source=${dataset_name}
fi

# model config
model_name="tiny"  # tiny mini small medium graphformer base large xlarge xxlarge

# train config
batch_size=128
workerCount=4
num_cpus=10
total_tokens=2e10
warmup_tokens=2e9
max_position_embeddings=1024
tie_word_embeddings=0
lr=3e-4
weight_decay=0.01
max_grad_norm=1
eps=1e-8
logit_adjust=0
pack_tokens=1
memory=40960
## deep-speed config
deepspeed_config="./examples/ds_config2_pt.json"
## tokenization config
attr_assignment="random"
ignored_off=0


# dataset config
if [ ${dataset_name} = "ogbn-proteins" ]
then
  samples_per_saving=10000000
  tokenization_config_file="ogbn_proteins_tokenization_config.json"
  ds_prefix="ogbn_proteins"
  # subgraph sampling config
  sampling_conf="ne_d9n1"
  # training config
  max_position_embeddings=256
elif [ ${dataset_name} = "ogbl-ppa" ]
then
  samples_per_saving=10000000  # 10,000,000 ~ 25% total train data 10000000
  tokenization_config_file="ogbl_ppa_tokenization_config.json"
  ds_prefix="ogbl_ppa"
  # subgraph sampling config
  sampling_conf="ee_d1n14"
  # training config
  max_position_embeddings=256
elif [ ${dataset_name} = "ogbg-molpcba" ]
then
  samples_per_saving=1000000
  tokenization_config_file="ogbg_molpcba_tokenization_config.json"
  ds_prefix="ogbg_molpcba"
  # subgraph sampling config
  sampling_conf="ns"
  # training config
  max_position_embeddings=1024
elif [ ${dataset_name} = "PCQM4Mv2" ]
then
  samples_per_saving=1000000
  tokenization_config_file="pcqm4m-v2_tokenization_config.json"
  ds_prefix="pcqm4m-v2"
  # subgraph sampling config
  sampling_conf="ns"
  # training config
  max_position_embeddings=1024
else
  echo "Using dataset ${dataset_name}, which is NOT implemented!!!"
fi


# model config
if [ ${model_name} = "tiny" ]
then
  hidden_size=128
  num_hidden_layers=2
  echo "In Tiny setting!"
elif [ ${model_name} = "mini" ]
then
  hidden_size=256
  num_hidden_layers=4
  echo "In Mini setting!"
elif [ ${model_name} = "small" ]
then
  hidden_size=512
  num_hidden_layers=4
  echo "In Small setting!"
elif [ ${model_name} = "medium" ]
then
  hidden_size=512
  num_hidden_layers=8
  echo "In Medium setting!"
elif [ ${model_name} = "graphformer" ]
then
  lr=2e-4
  hidden_size=768
  num_hidden_layers=12
  model_config_file="graphformer_model_config.json"

  optimization_config_file="graphformer_optimizer_config.json"
  if [ "${optimization_config_file}" = "" ]  # https://stackoverflow.com/a/13618376
  then
    optimization_config=""
  else
    optimization_config="./examples/${optimization_config_file}"
  fi
  suffix="_gf"
  echo "In Graphformer setting!"
elif [ ${model_name} = "base" ]
then
  hidden_size=768
  num_hidden_layers=12
  echo "In Base setting!"
elif [ ${model_name} = "large" ]
then
  hidden_size=1024
  num_hidden_layers=24
  echo "In Large setting!"
elif [ ${model_name} = "xlarge" ]
then
  hidden_size=1280
  num_hidden_layers=36
  echo "In XLarge setting!"
elif [ ${model_name} = "xxlarge" ]
then
  hidden_size=1600
  num_hidden_layers=48
  memory=40960
  echo "In XXLarge setting!"
else
  # model config
  hidden_size=768
  num_hidden_layers=12
  echo "Use customer setting of batch_size/hidden_size/num_hidden_layers"
fi


# env config
data_dir_prefix="./data"
output_dir_prefix="./exp/models"
tokenization_config="./examples/${tokenization_config_file}"
if [ "${model_config_file}" = "" ]  # https://stackoverflow.com/a/13618376
then
  model_config=""
else
  model_config="./examples/${model_config_file}"
fi


let batch_actual=batch_size*workerCount
data_dir="OGB"
output_dir="${ds_prefix}/pt_${sampling_conf}_h${hidden_size}_l${num_hidden_layers}_b${batch_actual}_mpe${max_position_embeddings}_tk${total_tokens}${suffix}"

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
  --ignored_off=${ignored_off}
  --batch_size=${batch_size}
  --pack_tokens=${pack_tokens}
  --num_workers=${num_cpus}
  --max_position_embeddings=${max_position_embeddings}
  --tie_word_embeddings=${tie_word_embeddings}
  --lr=${lr}
  --weight_decay=${weight_decay}
  --eps=${eps}
  --max_grad_norm=${max_grad_norm}
  --logit_adjust=${logit_adjust}
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
