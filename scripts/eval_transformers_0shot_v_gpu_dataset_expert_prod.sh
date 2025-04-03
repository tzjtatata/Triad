#! /bin/bash
model_path=$1
model_name=$2
v=$3
gpu=$4
dataset=$5
expert=$6
prod=$7
export CUDA_VISIBLE_DEVICES=$gpu
if [[ $CUDA_VISIBLE_DEVICES == '' ]]
then
    export CUDA_VISIBLE_DEVICES=0
fi

d_path=./evaluation/$dataset
ans_path=results/${model_name}/ans_${dataset}_${prod}_0shot_randomroi${expert}_${v}.json

echo eval $model_name in dataset $dataset mode ${eval_mode} and GPU $gpu

python llava_eval/evaluate.py --model-path $model_path --model-type transformers --ds $dataset --data_path $d_path --text_src "config:$v" --subset $prod --from question${expert}.jsonl --to $ans_path  --image_num 1 --qa_mode single_qa
python llava_eval/collect_acc.py $d_path/$ans_path > $model_path/result_${dataset}_${expert}_${v}_${prod}.csv