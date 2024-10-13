#!/bin/bash

dataset_name="ogbl-wikikg2"
true_valid=-1  # -1  -2

# i. data config
tokenizer_class="StackedGSTTokenizer"  # GSTTokenizer|StackedGSTTokenizer
tokenization_config_file="edge_lvl/ogbl_wikikg2_tokenization_config.json"
sampling_conf_sv="d1n14"

epochs=32
warmup_rate=0.3
epochs_warmup=$(bc <<<"scale=2;${epochs}*${warmup_rate}")

num_cpus=12  # -> consume too much cpu memory
batch_size=512
workerCount=8
# dataset-specific config
batch_size_eval=${batch_size}
epoch_per_eval=1
k_samplers=512  # num of samples per GPU for train data evaluation
max_position_embeddings_sv=1024
save_pred=0
eval_only=0
# directories
ds_prefix="ogbl_wikikg2"
mid_dir="202409/"

# ii. model config
model_name="base"  # tiny mini small medium base base24 base48 large xlarge xxlarge
stack_method="short"
stacked_feat_agg_method="gated"
hidden_act="gelu"  # llama -> `silu`, graphformer -> `gelu`
causal_attention=0
pretrain_cpt="pt_d1n14_h768_l12_b2048_mpe1024_tk2e10_gelu_nmlm_mrlinear_mtp0.8_0_0.2_lr3e-4_adp0.1_pdp0_edp0_mdp0_lsi0_gated_eulershort_wd0.1"
suffix_pt="_10percent_nr1_swgt_pt_b2048_tk2e10"


# iii. optimization config
lr=3e-5
min_lr=0
# dropout
attention_dropout=0.1
path_dropout=0.1
embed_dropout=0
mlp_dropout=0
layer_scale_init_val=1
# optimizer hps
weight_decay=0
beta2=0.99
betas="[0.9,${beta2}]"
eps=1e-10
max_grad_norm=1
use_ema=0
ema_decay=0.9999
## deep-speed config; set it to empty to enable native DDP training
deepspeed_config="./examples/ds_config2.json"

## iv. optimization objective
ntp_ratio=0
task_ratio=$(bc <<<"scale=2;1-${ntp_ratio}")
## dataset-specific supervised-task
task_level="edge"
problem_type="single_label_classification"
num_labels=2


#=================== BELOW FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============
model_name="tiny"
batch_size=128
workerCount=1
num_cpus=10
pretrain_cpt=""
suffix_pt="_no_pt"
#=================== ABOVE FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============


#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
#===================================== FT::BELOW TILL THE END ARE THE SAME FOR ALL DATASETS ============================
# distribute framework
#deepspeed_config=""
if [ "${deepspeed_config}" = "" ]
  then dist_config="_ddp"  # _ds|_ddp
  else dist_config="_ds"  # _ds|_ddp
fi
suffix_sv="${dist_config}_${hidden_act}_tv${true_valid}_lr${lr}_wr${warmup_rate}_norm${max_grad_norm}_eps${eps}_beta2${beta2}_${stack_method}_ema${use_ema}_${ema_decay}_adp${attention_dropout}_pdp${path_dropout}_edp${embed_dropout}_mdp${mlp_dropout}_lsi${layer_scale_init_val}_wd${weight_decay}${suffix_pt}"


# architectures
intermediate_size=0
num_attention_heads=0
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
elif [ ${model_name} = "base" ]
then
  hidden_size=768
  num_hidden_layers=12
  echo "In Base setting!"
elif [ ${model_name} = "base24" ]
then
  hidden_size=768
  num_hidden_layers=24
  echo "In Base24 setting!"
elif [ ${model_name} = "base48" ]
then
  hidden_size=768
  num_hidden_layers=48
  echo "In Base48 setting!"
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


let batch_size_actual=batch_size*workerCount
output_folder_raw="sv_${sampling_conf_sv}_h${hidden_size}_l${num_hidden_layers}_b${batch_size_actual}_mpe${max_position_embeddings_sv}_e${epochs}${suffix_sv}"

if [ "$pretrain_cpt" = "" ]
then
  output_folder="${output_folder_raw}"
else
  output_folder="pt2${output_folder_raw}"
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${mid_dir}${pretrain_cpt}"
fi

data_dir="OGB"
output_dir="${ds_prefix}/${mid_dir}${output_folder}"


raw_udf="
  --data_dir='${data_dir_prefix}/${data_dir}'
  --output_dir='${output_dir_prefix}/${output_dir}'
  --pretrain_cpt='${pretrain_cpt}'
  --dataset_name='${dataset_name}'
  --save_pred=${save_pred}
  --tokenizer_class='${tokenizer_class}'
  --tokenization_config='${tokenization_config}'
  --stack_method='${stack_method}'
  --batch_size=${batch_size}
  --batch_size_eval=${batch_size_eval}
  --num_workers=${num_cpus}
  --max_position_embeddings=${max_position_embeddings_sv}
  --lr=${lr}
  --min_lr=${min_lr}
  --eps=${eps}
  --betas='${betas}'
  --weight_decay=${weight_decay}
  --max_grad_norm=${max_grad_norm}
  --k_samplers=${k_samplers}
  --epochs=${epochs}
  --warmup_epochs=${epochs_warmup}
  --num_hidden_layers=${num_hidden_layers}
  --hidden_size=${hidden_size}
  --intermediate_size=${intermediate_size}
  --num_attention_heads=${num_attention_heads}
  --hidden_act='${hidden_act}'
  --stacked_feat_agg_method=${stacked_feat_agg_method}
  --task_level=${task_level}
  --num_labels=${num_labels}
  --problem_type=${problem_type}
  --loss_type='${loss_type}'
  --deepspeed_config='${deepspeed_config}'
  --epoch_per_eval=${epoch_per_eval}
  --eval_only=${eval_only}
  --true_valid=${true_valid}
  --causal_attention=${causal_attention}
  --attention_dropout=${attention_dropout}
  --path_dropout=${path_dropout}
  --embed_dropout=${embed_dropout}
  --mlp_dropout=${mlp_dropout}
  --layer_scale_init_value=${layer_scale_init_val}
  --use_ema=${use_ema}
  --ema_decay=${ema_decay}
"

udf=${raw_udf//$'\n'/}

echo ${udf}
echo ${pretrain_cpt}

deepspeed ./examples/train_supervised.py ${raw_udf}

echo $raw_udf
echo "Train and evaluation finished"
