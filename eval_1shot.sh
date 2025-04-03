exp_name=$1

get_model_path(){
    echo ./checkpoints/llava-siglip-so400m-onevision-qwen2-7b--${1}_1epoch_4x8x4_1e-5
}

model_path=`get_model_path $exp_name`

#bash eval_myriad2_v_gpu_dataset_expert_prod.sh $model_path $exp_name $v $gpu $dataset $expert $prod
bash scripts/eval_myriad2_1shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 0 mvtec_1shot _musc all
bash scripts/eval_myriad2_1shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v1 1 mvtec_1shot _musc all
bash scripts/eval_myriad2_1shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v2 2 mvtec_1shot _musc all
bash scripts/eval_myriad2_1shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v3 3 mvtec_1shot _musc all