#!/bin/bash

# dataset config
dataset_name="PCQM4Mv2"
dataset_source="PCQM4Mv2"  # molecule|PCQM4Mv2

# i. data config
pack_tokens=0
tokenizer_class="StackedGSTTokenizer"  # StackedGSTTokenizer  GSTTokenizer
tokenization_config_file="graph_lvl/pcqm4m-v2_tokenization_config.json"
sampling_conf="ns"

num_cpus=12
batch_size=1024  # 256/512 for pack_tokens=0, 64 for pk=1
workerCount=8
total_tokens=4e9  # 1e11  1e9
warmup_tokens=1e8  # 1e9  1e8
max_position_embeddings=1024
samples_per_saving=1000000

ds_prefix="pcqm4m-v2"
mid_dir="202409/"

# ii. model config
model_name="base24"  # tiny mini small medium base base24 base48 base64 large xlarge xxlarge
stack_method="short"
stacked_feat_agg_method="gated"  # gated|sum
tie_word_embeddings=0
hidden_act="gelu"  # llama -> `silu`, graphformer -> `gelu`

intermediate_size=0
num_attention_heads=0

# iii. optimization config
lr=3e-4
# dropout
attention_dropout=0.1
embed_dropout=0
path_dropout=0
mlp_dropout=0
layer_scale_init_val=0
# optimizer hps
weight_decay=0.1
max_grad_norm=5
eps=1e-8
use_ema=0
## deep-speed config; set it to empty to enable native DDP training
deepspeed_config="./examples/ds_config2_pt.json"

## iv. optimization objective
task_type="pretrain-mlm"  # pretrain  pretrain-mlm  pretrain-ltp  pretrain-euler

suffix="_${hidden_act}_3.3m_nmlm_mrlinear_mtp0.8_0_0.2_lr${lr}_adp${attention_dropout}_pdp${path_dropout}_edp${embed_dropout}_mdp${mlp_dropout}_lsi${layer_scale_init_val}_${stack_method}_${stacked_feat_agg_method}_wd${weight_decay}"
#===================================== ABOVE is config for producing our best results ==================================

#=================== BELOW FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============
#model_name="tiny"
#batch_size=128
#workerCount=1
#num_cpus=10
#total_tokens=1e9
#warmup_tokens=1e8
#=================== ABOVE FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============



#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
#===================================== PT:: BELOW TILL THE END ARE THE SAME FOR ALL DATASETS ===========================
# architectures
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
if [ "${model_config_file}" = "" ]  # https://stackoverflow.com/a/13618376
then
  model_config=""
else
  model_config="./examples/${model_config_file}"
fi


let batch_actual=batch_size*workerCount
data_dir="OGB"
output_dir="${ds_prefix}/${mid_dir}pt_${sampling_conf}_h${hidden_size}_l${num_hidden_layers}_b${batch_actual}_mpe${max_position_embeddings}_tk${total_tokens}${suffix}"

if [ "$pretrain_cpt" != "" ]
then
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${mid_dir}${pretrain_cpt}"
fi


raw_udf="
  --data_dir='${data_dir_prefix}/${data_dir}'
  --output_dir='${output_dir_prefix}/${output_dir}'
  --pretrain_cpt='${pretrain_cpt}'
  --dataset_name='${dataset_source}'
  --task_type='${task_type}'
  --tokenizer_class='${tokenizer_class}'
  --tokenization_config='${tokenization_config}'
  --stack_method='${stack_method}'
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
  --intermediate_size=${intermediate_size}
  --num_attention_heads=${num_attention_heads}
  --hidden_act='${hidden_act}'
  --stacked_feat_agg_method=${stacked_feat_agg_method}
  --samples_per_saving=${samples_per_saving}
  --deepspeed_config='${deepspeed_config}'
  --attention_dropout=${attention_dropout}
  --embed_dropout=${embed_dropout}
  --path_dropout=${path_dropout}
  --mlp_dropout=${mlp_dropout}
  --layer_scale_init_value=${layer_scale_init_val}
  --use_ema=${use_ema}
"

udf=${raw_udf//$'\n'/}

echo ${udf}

deepspeed ./examples/train_pretrain.py ${raw_udf}

echo $raw_udf
echo "Train and evaluation finished"
