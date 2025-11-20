#!/bin/bash

# i. data config
data_dir="TUDataset"
dataset_name="reddit_threads"

tokenizer_class="StackedGSTTokenizer"  # GSTTokenizer|StackedGSTTokenizer
token_cfg_dir="graph_lvl/"
token_cfg_file="reddit"

# ii. model config
model_type="graphgpt"
model_name="tiny"  # tiny mini small medium base base24 base48 large large48 xlarge xxlarge
stack_method="short"  # short|long
stacked_feat_agg_method="sum"  # sum|gated
hidden_act="gelu"  # llama -> `silu`
causal_attention=false
max_position_embeddings=1024

# ii.a model::dropout & lsi
attention_dropout=0
path_dropout=0
embed_dropout=0
mlp_dropout=0
layer_scale_init_val=0

# iii. training config
trial=1
epochs=16
#epochs_warmup=$(bc <<<"scale=2;${epochs}*0.3")
epochs_warmup=4.8
seed=${trial}

# iii.a training::training machines
num_cpus=4
workerCount=1

# iii.b training::training/eval data organization
batch_size=256
true_valid=-1  # -1  30000  10000  5000  0 => number of validation samples to use (-1 for all)
batch_size_eval=32
epoch_per_eval=1
k_samplers=256  # num of samples per GPU for train data evaluation

# iii.c training::eval/infer settings
save_pred=true
eval_only=false
save_hidden_states=false  # whether to infer hidden layer stats for downstream tasks

# iii.d training::directories
ds_prefix="reddit"
mid_dir="202511/"
pretrain_cpt=""
suffix_pt="_t${trial}"

# iii.e training::optimization config
lr=1e-4
min_lr=0
# optimizer hps
beta2=0.99
betas="[0.9,${beta2}]"
eps=1e-10
weight_decay=0
max_grad_norm=1
use_ema=false
ema_decay=0.9995
# deep-speed config; set it to empty to enable native DDP training
deepspeed_config="./examples/ds_config2.json"

## iii.f optimization objective
task_ratio=1
## dataset-specific supervised-task
task_level="graph"
problem_type="single_label_classification"
num_labels=2
loss_type=""
#===============================ABOVE section is task-specific==============================================


#=================== BELOW FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============
#model_name="tiny"
#batch_size=128
#workerCount=1
#num_cpus=10
#pretrain_cpt=""
#suffix_pt="_no_pt"
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

if [ "${use_ema}" = "false" ]
then ema=""
else ema="_ema${ema_decay}"
fi

if [ ${attention_dropout} -eq 0 ]
then adp=""
else adp="_adp${attention_dropout}"
fi

if [ ${path_dropout} -eq 0 ]
then pdp=""
else pdp="_pdp${path_dropout}"
fi

if [ ${embed_dropout} -eq 0 ]
then edp=""
else edp="_edp${embed_dropout}"
fi

if [ ${mlp_dropout} -eq 0 ]
then mdp=""
else mdp="_mdp${mlp_dropout}"
fi

if [ ${layer_scale_init_val} -eq 0 ]
then lsi=""
else lsi="_lsi${layer_scale_init_val}"
fi

suffix_sv="${dist_config}_tv${true_valid}_lr${lr}_norm${max_grad_norm}_eps${eps}_beta2${beta2}_${stack_method}${ema}${adp}${pdp}${edp}${mdp}${lsi}_wd${weight_decay}${suffix_pt}"

# architectures
intermediate_size=0
num_attention_heads=0

case "${model_name}" in
  "tiny")
    hidden_size=128
    num_hidden_layers=2
    echo "In Tiny setting!"
    ;;
  "tiny6")
    hidden_size=128
    num_hidden_layers=6
    intermediate_size=512
    num_attention_heads=4
    echo "In Tiny6 setting!"
    ;;
  "mini")
    hidden_size=256
    num_hidden_layers=4
    echo "In Mini setting!"
    ;;
  "small")
    hidden_size=512
    num_hidden_layers=4
    echo "In Small setting!"
    ;;
  "small12")
    hidden_size=384
    intermediate_size=384  # intermediate_size = 1*hidden_size
    num_attention_heads=12  # d=32 per head
    num_hidden_layers=12
    echo "In Small12 setting!"
    ;;
  "medium")
    hidden_size=512
    num_hidden_layers=8
    echo "In Medium setting!"
    ;;
  "base")
    hidden_size=768
    num_hidden_layers=12
    echo "In Base setting!"
    ;;
  "base24")
    hidden_size=768
    num_hidden_layers=24
    echo "In Base24 setting!"
    ;;
  "base48")
    hidden_size=768
    num_hidden_layers=48
    echo "In Base48 setting!"
    ;;
  "large")
    hidden_size=1024
    num_hidden_layers=24
    echo "In Large setting!"
    ;;
  "large48")
    hidden_size=1024
    num_hidden_layers=48
    echo "In Large48 setting!"
    ;;
  "xlarge")
    hidden_size=1280
    num_hidden_layers=36
    echo "In XLarge setting!"
    ;;
  "xlarge48")
    hidden_size=1280
    num_hidden_layers=48
    echo "In XLarge48 setting!"
    ;;
  "xxlarge")
    hidden_size=1600
    num_hidden_layers=48
    echo "In XXLarge setting!"
    ;;
  *)
    # Default case for any other model name
    hidden_size=768
    num_hidden_layers=12
    echo "Use customer setting of batch_size/hidden_size/num_hidden_layers"
    ;;
esac


# env config
data_dir_prefix="./data"
output_dir_prefix="./exp/models"


batch_size_actual=$((batch_size * workerCount))
output_folder_raw="sv_h${hidden_size}_l${num_hidden_layers}_b${batch_size_actual}_mpe${max_position_embeddings}_e${epochs}${suffix_sv}"

if [[ $pretrain_cpt == "" ]]
then
  output_folder="${output_folder_raw}"
else
  output_folder="pt2${output_folder_raw}"
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${mid_dir}${pretrain_cpt}"
fi

output_dir="${ds_prefix}/${mid_dir}${output_folder}"


raw_udf="
  --tokenization.tokenizer_class='${tokenizer_class}'
  --tokenization.data.data_dir='${data_dir_prefix}/${data_dir}'
  --tokenization.data.dataset='${dataset_name}'
  --training.deepspeed_conf_file='${deepspeed_config}'
  --training.output_dir='${output_dir_prefix}/${output_dir}'
  --training.pretrain_cpt='${pretrain_cpt}'
  --training.task_type='${task_level}'
  --training.batch_size=${batch_size}
  --training.batch_size_eval=${batch_size_eval}
  --training.max_length=${max_position_embeddings}
  --training.num_workers=${num_cpus}
  --training.optimizer.lr=${lr}
  --training.optimizer.min_lr=${min_lr}
  --training.optimizer.betas=${betas}
  --training.optimizer.eps=${eps}
  --training.optimizer.weight_decay=${weight_decay}
  --training.optimizer.max_grad_norm=${max_grad_norm}
  --training.optimizer.use_ema=${use_ema}
  --training.optimizer.ema_decay=${ema_decay}
  --training.schedule.epochs=${epochs}
  --training.schedule.warmup_epochs=${epochs_warmup}
  --training.finetune.seed=${seed}
  --training.ft_eval.save_pred=${save_pred}
  --training.ft_eval.save_hidden_states=${save_hidden_states}
  --training.ft_eval.k_samplers=${k_samplers}
  --training.ft_eval.epoch_per_eval=${epoch_per_eval}
  --training.ft_eval.eval_only=${eval_only}
  --training.ft_eval.true_valid=${true_valid}
  --model.model_type='${model_type}'
  --model.max_position_embeddings=${max_position_embeddings}
  --model.num_hidden_layers=${num_hidden_layers}
  --model.hidden_size=${hidden_size}
  --model.intermediate_size=${intermediate_size}
  --model.num_attention_heads=${num_attention_heads}
  --model.hidden_act='${hidden_act}'
  --model.causal_attention=${causal_attention}
  --model.graph_input.stacked_feat_agg_method='${stacked_feat_agg_method}'
  --model.graph_input.stack_method='${stack_method}'
  --model.ft_head.num_labels=${num_labels}
  --model.ft_head.problem_type='${problem_type}'
  --model.ft_head.loss_type='${loss_type}'
  --model.ft_head.task_ratio=${task_ratio}
  --model.dropout_settings.attention_dropout=${attention_dropout}
  --model.dropout_settings.path_dropout=${path_dropout}
  --model.dropout_settings.embed_dropout=${embed_dropout}
  --model.dropout_settings.mlp_dropout=${mlp_dropout}
  --model.layer_scale_init_value=${layer_scale_init_val}
"

udf=${raw_udf//$'\n'/}
udf=${udf//--/}

echo ${udf}
echo ${pretrain_cpt}

deepspeed ./examples/train_supervised.py tokenization=${token_cfg_dir}${token_cfg_file} ${udf}

echo $raw_udf
echo "Train and evaluation finished"
