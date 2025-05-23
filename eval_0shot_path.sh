model_path=$1

get_exp_name(){
    fname=`basename $1`
    prefix="llava-siglip-so400m-onevision-qwen2-7b--"
    suffix="_1epoch_4x8x4_1e-5"
    echo $fname | sed -n 's/'"$prefix"'//gp' | sed -n 's/'"$suffix"'//gp'
}
exp_name=`get_exp_name $model_path`

if [[ $exp_name == "" ]]
then
    exp_name=`basename $model_path`
fi

echo $exp_name

#bash eval_triad_v_gpu_dataset_expert_prod.sh $model_path $exp_name $v $gpu $dataset $expert $prod
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 0 mvtec _musc all
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v1 1 mvtec _musc all
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v2 0 mvtec _musc all
bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v3 1 mvtec _musc all

#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 0 WFDD _musc all
#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name prod_v1 1 WFDD _musc all
#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name v0 2 pcb_bank _musc pcb6,pcb7
#bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh $model_path $exp_name prod_v1 3 pcb_bank _musc pcb6,pcb7
