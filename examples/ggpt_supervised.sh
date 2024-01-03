#!/bin/bash

#### below useless hp
dropout=0
use_mlp=0
gradient_accumulation_steps=1
memory=20480
freeze=-1
# sample graphs for train proportional to their number of eulerian paths/num_nodes
with_prob=0
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
dataset_name="ogbn-proteins"  # PCQM4Mv2  ogbg-molpcba  ogbl-ppa  ogbn-proteins
dataset_source=""  # molecule  PCQM4Mv2
if [ "${dataset_source}" = "" ]
then
  dataset_source=${dataset_name}
fi

# model config
model_name="mini"  # tiny mini small medium graphformer base large xlarge xxlarge

# train config
lr=1e-4
beta2=0.99
betas="[0.9,${beta2}]"
eps=1e-10
weight_decay=0
max_grad_norm=1

epochs_actual=32
epochs_warmup_actual=$(bc <<<"scale=2;${epochs_actual}*0.3")
batch_size_sv=256
workerCount=4
num_cpus=10
seed=42
# deep-speed config
deepspeed_config="./examples/ds_config2.json"
# others
suffix_sv=""

# supervised-task config
if [ ${dataset_name} = "ogbn-proteins" ]
then
  task_level="node"
  problem_type="multi_label_classification"
  num_labels=112
elif [ ${dataset_name} = "ogbl-ppa" ]
then
  task_level="edge"
  problem_type="single_label_classification"
  num_labels=2
elif [ ${dataset_name} = "ogbg-molpcba" ]
then
  task_level="graph"
  problem_type="multi_label_classification"
  num_labels=128
elif [ ${dataset_name} = "PCQM4Mv2" ]
then
  task_level="graph"
  problem_type="regression"
  num_labels=1
  loss_type="l1"
else
  echo "Using dataset ${dataset_name}, which is NOT implemented!!!"
fi


# dataset config
if [ ${dataset_name} = "ogbn-proteins" ]
then
  samples_per_saving=86619  # 1000000
  samples_per_eval=0  # 40000
  k_samplers=256  # 2^12=4096  2^14=16384  2^16=65536  2^18=262144
  tokenization_config_file="ogbn_proteins_tokenization_config.json"
  model_config_file="dropout_model_config.json"
  ds_prefix="ogbn_proteins"
  # train config
  sampling_conf_sv="d9n1"
  max_position_embeddings_sv=256
  # save
  save_pred=0
elif [ ${dataset_name} = "ogbl-ppa" ]
then
  samples_per_saving=10000000  # 10,000,000 ~ 25% total train data
  samples_per_eval=5000000
  k_samplers=1024  # 2^12=4096  2^13=8192  2^14=16384  2^16=65536  2^18=262144
  tokenization_config_file="ogbl_ppa_tokenization_config.json"
  ds_prefix="ogbl_ppa"
  # train config
  sampling_conf_sv="d1n14"
  max_position_embeddings_sv=256
elif [ ${dataset_name} = "ogbg-molpcba" ]
then
  samples_per_saving=350343  # 350343/43793/43793 -> train/valid/test
  samples_per_eval=0  # 300000
  k_samplers=1024  # 2^12=4096  2^13=8192  2^14=16384  2^16=65536  2^18=262144
  tokenization_config_file="ogbg_molpcba_tokenization_config.json"
  model_config_file="dropout_model_config.json"
  ds_prefix="ogbg_molpcba"
  # train config
  sampling_conf_sv="ns"
  max_position_embeddings_sv=1024
  # save
  save_pred=0
elif [ ${dataset_name} = "PCQM4Mv2" ]
then
  samples_per_saving=0  # 3378606/73545/36773 -> train/valid/test
  samples_per_eval=0  # 2000000
  k_samplers=512  # 2^12=4096  2^13=8192  2^14=16384  2^16=65536  2^18=262144
  tokenization_config_file="pcqm4m-v2_tokenization_config.json"
  # model_config_file="pcqm4m-v2_model_config.json"
  ds_prefix="pcqm4m-v2"
  memory=40960
  # train config
  sampling_conf_sv="ns"
  max_position_embeddings_sv=1024
  # other
  true_valid=5000  # 128  512  4096  5000
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
  hidden_size=768
  num_hidden_layers=12
  model_config_file="graphformer_model_config.json"
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
  memory=71680  # 1024*7*10  needs huge memory to offload
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


# pre-train ckp:: dataset-specific config
if [ ${dataset_name} = "ogbn-proteins" ]
then
  pretrain_cpt="pt_ne_d9n1_h${hidden_size}_l${num_hidden_layers}_b1024_mpe256_tk2e10"
elif [ ${dataset_name} = "ogbl-ppa" ]
then
  pretrain_cpt="pt_ee_d1n14_h${hidden_size}_l${num_hidden_layers}_b1600_mpe256_tk2e10"
elif [ ${dataset_name} = "ogbg-molpcba" ]
then
  pretrain_cpt="pt_ns_h${hidden_size}_l${num_hidden_layers}_b1024_mpe1024_tk2e10"
elif [ ${dataset_name} = "PCQM4Mv2" ]
then
  pretrain_cpt="pt_ns_h${hidden_size}_l${num_hidden_layers}_b512_mpe1024_tk2e10"
else
  echo "Using dataset ${dataset_name}, which is NOT implemented!!!"
fi


let batch_size_actual=batch_size_sv*workerCount*gradient_accumulation_steps
output_folder_raw="sv_${sampling_conf_sv}_h${hidden_size}_l${num_hidden_layers}_b${batch_size_actual}_mpe${max_position_embeddings_sv}_e${epochs_actual}${suffix_sv}"

if [ "$pretrain_cpt" = "" ]
then
  output_folder="${output_folder_raw}"
else
  output_folder="pt2${output_folder_raw}"
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${pretrain_cpt}"
fi

data_dir="OGB"
output_dir="${ds_prefix}/${output_folder}"


raw_udf="
  --data_dir='${data_dir_prefix}/${data_dir}'
  --output_dir='${output_dir_prefix}/${output_dir}'
  --pretrain_cpt='${pretrain_cpt}'
  --dataset_name='${dataset_name}'
  --with_prob=${with_prob}
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
  --epochs=${epochs_actual}
  --warmup_epochs=${epochs_warmup_actual}
  --model_config='${model_config}'
  --num_hidden_layers=${num_hidden_layers}
  --hidden_size=${hidden_size}
  --task_level=${task_level}
  --num_labels=${num_labels}
  --mlp='[${mlp}]'
  --dropout=${dropout}
  --problem_type=${problem_type}
  --loss_type=${loss_type}
  --samples_per_eval=${samples_per_eval}
  --deepspeed_config='${deepspeed_config}'
  --gradient_accumulation_steps=${gradient_accumulation_steps}
  --optimization_config='${optimization_config}'
  --eval_only=${eval_only}
  --seed=${seed}
  --true_valid=${true_valid}
"

udf=${raw_udf//$'\n'/}

echo ${udf}
echo ${pretrain_cpt}

deepspeed ./examples/train_supervised.py ${raw_udf}

echo $raw_udf
echo "Train and evaluation finished"
