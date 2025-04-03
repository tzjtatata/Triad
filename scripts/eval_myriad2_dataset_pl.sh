#! /bin/bash
model_path=$1
model_name=$2
gpu=$3
dataset=$4
p=$5
prod=all
export CUDA_VISIBLE_DEVICES=$gpu
if [[ $CUDA_VISIBLE_DEVICES == '' ]]
then
    export CUDA_VISIBLE_DEVICES=0
fi

d_path=./evaluation/$dataset
ans_path=results/${model_name}/ans_${dataset}_${prod}_0shot_p_${p}.json

echo eval $model_name in dataset $dataset mode ${p} and GPU $gpu

python llava_eval/evaluate.py --model-path $model_path --model-type myriad2 --ds $dataset --data_path $d_path --text_src "dataset" --subset $prod --from $p --to $ans_path --qa_mode single_qa
python llava_eval/collect_acc.py $d_path/$ans_path > $model_path/result_${dataset}_${prod}_p_${p}.csv