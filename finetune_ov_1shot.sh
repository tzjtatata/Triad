export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4000

LLM_VERSION_OV="/hdd/ysh/pretrained_models/models--lmms-lab--llava-onevision-qwen2-7b-ov/"
LLM_VERSION_SI="/hdd/ysh/pretrained_models/models--lmms-lab--llava-onevision-qwen2-7b-si/"
#VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"

get_model_path(){
    echo ./checkpoints/llava-siglip-so400m-onevision-qwen2-7b--${1}_1epoch_4x8x4_1e-5
}

exp_name=llava_ov_1shot_reproduce_si_12k_realiad_10k_v1_coco_4k_1shot_mvtec_loco_clgen_llava_onevision_qwen_ov_aft_fix_32768_randomroi_new_new_code_test
model_path=`get_model_path ${exp_name}`
log_name=llava_train_$(date "+%Y%m%d_%H%M%S").log
echo Will save to $model_path

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="4" --nnodes="1" --master_addr="localhost" --master_port="29600" \
    LLaVA-NeXT/llava/train/train_mem.py \
    --deepspeed LLaVA-NeXT/scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION_OV} \
    --version ${PROMPT_VERSION} \
    --data_path="{llava_extension_data/multiImg_sharegpt4v_coco_train_db_4k_bbox_v2.json,llava_extension_data/multiImg_visa/visa_llama3_comparison_only_diff_anomaly_and_normal_whl_fixed_mask.json,llava_extension_data/multiImg_visa/visa_llama3_comparison_only_diff_only_anomaly_whl_fixed_mask.json,llava_extension_data/llava_onevision_si_mixed_12k.json,llava_extension_data/realiad_db_new_10k_mask_v1.json,llava_extension_data/realiad_reasoning_1_db_139_comparison_swap_choice.json,llava_extension_data/realiad_reasoning_1_db_139_swap_choice.json,llava_extension_data/realiad_reasoning_2_db_167_comparison_swap_choice.json,llava_extension_data/realiad_reasoning_2_db_167_swap_choice.json,lyz_analysis/ND2Checklist_Text_mvtec_12.json,lyz_analysis/PP2Checklist_Text_mvtec-realiad_17.json,lyz_analysis/mvtec_loco_1shot_checklistGen_v4_details_bbox-aug-384px_400.json,lyz_analysis/mvtec_loco_1shot_checklistGen_v4_selection_bbox-aug-384px_400.json,lyz_analysis/textual_trainv4-type2_mvtec_3class_pp_240.json,lyz_analysis/textual_trainv4-type1_mvtec_3class_pp_240.json,lyz_analysis/textual_trainv4-type1_mvtec_cable_pp_41.json,lyz_analysis/textual_trainv4-type2_mvtec_cable_pp_41.json}" \
    --image_folder /home/m/lyz/data/llava_extension_data \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio randomroi \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type "spatial_avgpool_auto_unpad_add_newl" \
    --multi_image_pad_threshold 3 \
    --bf16 True \
    --run_name $exp_name \
    --output_dir $model_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --attn_implementation sdpa \
    --dataloader_drop_last True \
    --report_to tensorboard 2>&1 | tee tmp_log/llava_train_$(date "+%Y%m%d_%H%M%S").log
