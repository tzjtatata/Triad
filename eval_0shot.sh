exp_name=$1

get_model_path(){
    echo ./checkpoints/llava-siglip-so400m-onevision-qwen2-7b--${1}_1epoch_4x8x4_1e-5
}

model_path=`get_model_path $exp_name`

#bash eval_triad_v_gpu_dataset_expert_prod.sh $model_path $exp_name $v $gpu $dataset $expert $prod
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 0 mvtec _musc all
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v1 1 mvtec _musc all
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v2 0 mvtec _musc all
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v3 1 mvtec _musc all

#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 0 WFDD _musc all
#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name prod_v1 1 WFDD _musc all
#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 2 pcb_bank _musc pcb6,pcb7
#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name prod_v1 3 pcb_bank _musc pcb6,pcb7
