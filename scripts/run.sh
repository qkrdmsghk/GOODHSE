#!/bin/bash

export device=$1
export dataset=$2
export project_name='240516_reproduce'

if [ $dataset = 'icassay' ]
then
    export DATASET='drugood_lbap_core_ic50_assay'
elif [ $dataset = 'icscaffold' ]
then
    export DATASET='drugood_lbap_core_ic50_scaffold'
elif [ $dataset = 'icsize' ]
then
    export DATASET='drugood_lbap_core_ic50_size'
elif [ $dataset = 'ecassay' ]
then
    export DATASET='drugood_lbap_core_ec50_assay'
elif [ $dataset = 'ecscaffold' ]
then
    export DATASET='drugood_lbap_core_ec50_scaffold'
elif [ $dataset = 'ecsize' ]
then
    export DATASET='drugood_lbap_core_ec50_size'
fi

export BATCH_SIZE=(32 128)
export EI_LR=(1e-3)
export IL_LR=(1e-4 5e-4)
export EMB_DIM=(128 300)
export EI_NUM_LAYERS=(1)
export IL_NUM_LAYERS=(4 5)
export EARLY_STOPPING=(20)
export DROPOUT=(0.5)
export IRM_P=(0.01 0.001)
export NUM_HIER=(3)
export EI_LAST_HIER=(2)
export EI_EPOCHS=(20)
export IL_EPOCHS=(100)
export R=(0.8 0.6)
export model='gin'
export pooling='sum'
export eval_metric='auc'
export num_envs=-1
export il_cls='linear'
export irm_opt='ednil_EI_last_hier_ed_contrast_me_consist_stochastic_240516'


for batch_size in "${BATCH_SIZE[@]}"; do
for ei_lr in "${EI_LR[@]}"; do
for il_lr in "${IL_LR[@]}"; do
for emb_dim in "${EMB_DIM[@]}"; do
for r in "${R[@]}"; do
for ei_num_layers in "${EI_NUM_LAYERS[@]}"; do
for il_num_layers in "${IL_NUM_LAYERS[@]}"; do
for early_stopping in "${EARLY_STOPPING[@]}"; do
for dropout in "${DROPOUT[@]}"; do
for irm_p in "${IRM_P[@]}"; do
for num_hierarchy in "${NUM_HIER[@]}"; do
for ei_last_hierarchy in "${EI_LAST_HIER[@]}"; do
for ei_epochs in "${EI_EPOCHS[@]}"; do
for il_epochs in "${IL_EPOCHS[@]}"; do
    export log_dir=logs/$irm_opt/
    export output_file=$dataset-bs_$batch_size-emb_$emb_dim-ratio_$r-illr_$il_lr-ilnl_$il_num_layers-pt_$early_stopping-dr_$dropout-irm_p_$irm_p-nh_$num_hierarchy-ei_$ei_last_hierarchy-cls-$il_cls-pool_$pooling-model_$model*
    echo $log_dir$output_file
    if [[ ! -f $log_dir$output_file ]]; then
        echo "start training $output_file"
        python main_EM_ednil_hierarchical_consistent_stochastic.py \
        --wbproject_name $project_name \
        --device $device \
        --dataset $DATASET \
        --batch_size $batch_size \
        --EI_lr $ei_lr \
        --IL_lr $il_lr \
        --emb_dim $emb_dim \
        --r $r \
        --model $model \
        --pooling $pooling \
        --EI_num_layers $ei_num_layers \
        --IL_num_layers $il_num_layers \
        --early_stopping $early_stopping \
        --dropout $dropout \
        --eval_metric $eval_metric \
        --num_envs $num_envs \
        --irm_p $irm_p \
        --irm_opt $irm_opt \
        --EI_epochs $ei_epochs \
        --IL_epochs $il_epochs \
        --num_hierarchy $num_hierarchy \
        --ei_last_hierarchy $ei_last_hierarchy \
        --il_last_hierarchy \
        --il_cls $il_cls \
        --commit $output_file 
    fi
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done