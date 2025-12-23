#!/bin/bash

# i. data config
data_dir="OGB"
dataset_source="PCQM4Mv2"  # molecule|PCQM4Mv2
tokenizer_class="StackedGSTTokenizer"  # StackedGSTTokenizer  GSTTokenizer
token_cfg_dir="graph_lvl/"
token_cfg_file="pcqm4m-v2_2d"

# ii. model config
model_type="graphgpt"
model_name="base"  # tiny mini small medium base base24 base48 base64 large xlarge xxlarge
stack_method="short"
stacked_feat_agg_method="sum"  # gated|sum
max_position_embeddings=1024

# ii.a model::dropout & lsi
attention_dropout=0.1
path_dropout=0
embed_dropout=0
mlp_dropout=0
layer_scale_init_val=0

# iii. training config
trial=1
pack_tokens=0
batch_size=256

# iii.a training::training machines
workerCount=1
num_cpus=12

# iii.b training::schedule
total_tokens=1e9  # 1e11  1e9
warmup_tokens=1e8  # 1e9  1e8
steps_per_saving=1000  # From samples_per_saving=1000000 -> (1000000/(64*16) ≈ 976)
let samples_per_saving=steps_per_saving*batch_size*workerCount
logging_steps=100

# iii.c training::eval/infer settings
valid_percent=0.1
pt_eval_only=false
do_infer=false

# iii.d training::directories
ds_prefix="pcqm4m-v2"
mid_dir="202511/"
pretrain_cpt=""

# iii.e training::optimization config
lr=3e-4
# optimizer hps
weight_decay=0.1
max_grad_norm=1
eps=1e-8
use_ema=0
## deep-speed config; set it to empty to enable native DDP training
deepspeed_config="./examples/ds_config2_pt.json"

# iii.f training::optimization objective
task_type="pretrain-mlm"  # pretrain-cl
dlm_wgt=false
focal_gamma=0

# iii.g others
tot_samples=10000  # tot_samples sampled for evaluating average eulerian path length

# iv. generation config
do_generation=false
gen_alg="maskgit_plus"  # origin maskgit_plus topk_margin entropy
parallel_gen=false  # whether to parallel batch generation: slow when tested in spice-circuit dataset
#===================================== ABOVE section is task-specific ==================================

#=================== BELOW FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============
#model_name="tiny"
#batch_size=128
#workerCount=1
#num_cpus=4
#total_tokens=1e9
#warmup_tokens=1e8
#pretrain_cpt=""
#tot_samples=100
#=================== ABOVE FOR SINGLE GPU TESTING, COMMENT OUT IN NORMAL TRAINING ==============



#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
#===================================== PT:: BELOW TILL THE END ARE THE SAME FOR ALL DATASETS ===========================
if [ "${dlm_wgt}" == "true" ]
then
  loss_obj="dlm"
else
  loss_obj="mlm"
fi

if [ "${task_type}" == "pretrain-cl" ]
then
  pt_obj="cl"
else
  pt_obj="gen"
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

suffix="_t${trial}_vp${valid_percent}_${pt_obj}_${loss_obj}_lr${lr}${adp}${pdp}${edp}${mdp}${lsi}_${stacked_feat_agg_method}_${stack_method}_wd${weight_decay}"

# env config
data_dir_prefix="./data"
output_dir_prefix="./exp/models"

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
    # memory=71680  # Note: This variable is unused in the script
    echo "In XXLarge setting!"
    ;;
  *)
    # Default case for any other model name
    hidden_size=768
    num_hidden_layers=12
    echo "Use customer setting of batch_size/hidden_size/num_hidden_layers"
    ;;
esac


batch_actual=$((batch_size * workerCount))

output_dir="${ds_prefix}/${mid_dir}pt_h${hidden_size}_l${num_hidden_layers}_tk${total_tokens}_b${batch_actual}_mpe${max_position_embeddings}${suffix}"

if [[ $pretrain_cpt != "" ]]
then
  pretrain_cpt="${output_dir_prefix}/${ds_prefix}/${mid_dir}${pretrain_cpt}"
fi


raw_udf="
  --tokenization.tokenizer_class='${tokenizer_class}'
  --tokenization.data.data_dir='${data_dir_prefix}/${data_dir}'
  --tokenization.data.dataset='${dataset_source}'
  --training.deepspeed_conf_file='${deepspeed_config}'
  --training.output_dir='${output_dir_prefix}/${output_dir}'
  --training.pretrain_cpt='${pretrain_cpt}'
  --training.task_type='${task_type}'
  --training.batch_size=${batch_size}
  --training.pack_tokens=${pack_tokens}
  --training.num_workers=${num_cpus}
  --training.optimizer.lr=${lr}
  --training.optimizer.weight_decay=${weight_decay}
  --training.optimizer.eps=${eps}
  --training.optimizer.max_grad_norm=${max_grad_norm}
  --training.optimizer.use_ema=${use_ema}
  --training.pretrain_mlm.dlm_wgt=${dlm_wgt}
  --training.schedule.total_tokens=${total_tokens}
  --training.schedule.warmup_tokens=${warmup_tokens}
  --training.schedule.samples_per_saving=${samples_per_saving}
  --training.schedule.logging_steps=${logging_steps}
  --training.valid_percent=${valid_percent}
  --training.do_generation=${do_generation}
  --training.pt_eval_only=${pt_eval_only}
  --training.do_infer=${do_infer}
  --training.tot_samples=${tot_samples}
  --training.focal_gamma=${focal_gamma}
  --model.model_type='${model_type}'
  --model.max_position_embeddings=${max_position_embeddings}
  --model.num_hidden_layers=${num_hidden_layers}
  --model.hidden_size=${hidden_size}
  --model.intermediate_size=${intermediate_size}
  --model.num_attention_heads=${num_attention_heads}
  --model.graph_input.stacked_feat_agg_method='${stacked_feat_agg_method}'
  --model.graph_input.stack_method='${stack_method}'
  --model.dropout_settings.attention_dropout=${attention_dropout}
  --model.dropout_settings.path_dropout=${path_dropout}
  --model.dropout_settings.embed_dropout=${embed_dropout}
  --model.dropout_settings.mlp_dropout=${mlp_dropout}
  --model.layer_scale_init_value=${layer_scale_init_val}
  --generation.alg=${gen_alg}
  --generation.parallel_gen=${parallel_gen}
"

udf=${raw_udf//$'\n'/}
udf=${udf//--/}

echo ${udf}

deepspeed ./examples/train_pretrain.py tokenization=${token_cfg_dir}${token_cfg_file} ${udf}

echo $raw_udf
echo "Train and evaluation finished"
