"""
unified evaluate scripts for myriad2 and transformers models on IAD benchmarks
"""
import sys
sys.path.append("./LLaVA-NeXT/")

import json
import jsonlines
import time
import math
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse

from evaluation_config import *

from PIL import Image
from copy import deepcopy
from simple_model_worker import SimpleModelWrapper,TransformersModelWrapper

def make_convs(convs,img_num=0):
    '''
    make_convs -> dict convert a list of convs to dict to meet limit of llava train data
    eg:
    [ q1, a1, q2, a2, ...] -> [{"from": "human", "value": q1}, {"from": "gpt", "value": a1}, ...]
    '''
    data_convs=[]
    roles=["human","gpt"]
    for i,c in enumerate(convs):
        if i==0:
            if "<image>" not in c:
                c="".join(["<image>\n"]*img_num)+c
        data_convs.append({"from":roles[i%2],"value":c})
    return data_convs

def make_train_data_json(data_id,img_ps,convs,img_p=None,masks=None,bboxes=None,**kwargs):
    '''
    make_train_data_json -> dict produce dict in format of llava train data
    data_id -> str : corresponding "id" in llava train data
    img_ps -> list[str] : list of image paths will be store in "src", note if only 1 image, "image" will replace "src"
    convs -> list[str] : list of conversations in format of [ q1, a1, q2, a2, ...], will be auto converted to llava train data format
    img_p -> str : image path that saved in "image", will use "src"[0] if set to None
    masks -> list[str] : mask paths that saved in "masks"
    bboxes -> list[list[list[int]]] : bboxes saved in "bboxes" for every image, every bbox, every axis
    **kwargs -> Any : any other kwargs will auto convert to key-value in data dict
    '''
    data={}
    data["id"]=data_id
    if type(img_ps) is str:
        img_ps=[img_ps]
    if len(img_ps)>1:
        data["src"]=img_ps
    if img_p is None:
        data["image"]=img_ps[0]
    else:
        data["image"]=img_p
    data["image_num"]=img_num=len(img_ps)
    data["conversations"]=make_convs(convs,img_num)
    if masks is not None:
        if type(masks) is list and len(masks)==1:
            masks=masks[0]
        data["mask"]=masks
    if bboxes is not None:
        data["bbox"]=bboxes
    data.update(kwargs)
    return data

def data_read(data_path):
    '''
    data_read -> list[dict] used to read data from question.jsonl or other json file storing eval data annotations, 
    current support llava train data format and question.jsonl format, after read data should be in llava train data format
    data_path -> str : the full path of the eval data file you want to read
    '''
    data_dir=os.path.dirname(data_path)
    dataset_name=os.path.basename(data_dir)
    all_datas=[]
    if data_path.endswith(".jsonl"): 
        with jsonlines.open(data_path) as f:
            jsonl_datas=list(f)
        for i in jsonl_datas:
            question_id=i["question_id"]
            img_p=os.path.join("imgs",i["image"])
            text=i["text"]
            ref=i.get("reference",None)
            if "sub_dataset" in i:
                subset=i["sub_dataset"] # in old formation it is sub_dataset
            else: # try to guess a subset
                default_subset=i["origin_path"].split("/")[0]
                if default_subset==dataset_name:
                    default_subset=i["origin_path"].split("/")[1]
                subset=i.get("subset", default_subset)
            gt=i.get("gt",None)
            origin_path=i["origin_path"]
            mask_p=i.get("mask",None)
            if mask_p is not None:
                mask_p=os.path.join(mask_p)
            bbox=i.get("bbox",None)
            if ref is not None:
                ref=os.path.join("mvtec",ref)
                img_ps=[img_p,ref]
            else:
                img_ps=[img_p]
            all_datas.append(make_train_data_json(question_id,img_ps,[text,""],img_ps[0],mask_p,bbox,subset=subset,gt=gt,origin_path=origin_path))
    else: # llava train data formation, still need "subset" "gt" "origin_path" just for unified mid format
        with open(data_path) as f:
            all_datas=json.load(f)
        for i in all_datas:
            if "subset" not in i:
                i["subset"]=i["image"].split("/")[1]
            if "gt" not in i:
                i["gt"]=0 if "/good/" in i["image"] else 1
            if "origin_path" not in i:
                i["origin_path"]=i["image"]
            if "bbox" not in i:
                i["bbox"]=None
            if "mask" not in i:
                i["mask"]=None
    return all_datas
    
def data_write(ans,out_path,format="json",mode="w"):
    '''
    data_write -> None write answers to json or jsonl file for mid storage, generally jsonl for mid temp storage and json for final result.
    ans -> list[dict] : all answers you want to save
    out_path -> str : full path of answer file path
    format -> ["json","jsonl"] : the out file format
    mode -> ["a","w"] : write mode of file
    '''
    ans_dir=os.path.dirname(out_path)
    os.makedirs(ans_dir,exist_ok=True)
    if format=="jsonl":
        out_path=out_path.replace(".json",".jsonl")
        with jsonlines.open(out_path,mode) as f:
            f.write(ans)
    else:
        out_path=out_path.replace(".jsonl",".json")
        with open(out_path,mode) as f:
            json.dump(ans,f,indent=4)
            
def read_mask(mask_p):
    '''
    read_mask -> np.array : read mask from mask_p support image and numpy mats
    '''
    if mask_p.endswith((".png",".jpg",".PNG",".JPG")):
        mask=Image.open(mask_p).convert('L')
    elif mask_p.endswith((".npz",".npy")):
        mask=np.load(mask_p)
        if isinstance(mask,dict):
            mask=mask["anomaly_map"]
    return mask
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-type", type=str, default='myriad2', choices=['myriad2', 'transformers', 'sglang'])
    parser.add_argument("--use-flash-attn", action="store_true")
    # dataset config
    parser.add_argument("--ds", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./evaluate/", help="DO NOT add / behind data_path if you want directly the dataset dir")
    parser.add_argument("--text_src", type=str, required=True)# choices=['dataset', 'config:version', 'input:prompt']
    parser.add_argument("--subset", type=str, default="all") # name of subsets in eval data, separate with ',' if multiple subsets
    parser.add_argument("--from", type=str, default='question.jsonl', dest='question_file')
    parser.add_argument("--to", type=str, default='answer.json', dest='answer_file')
    parser.add_argument("--start", type=int, default=0) # start from mid, for resuming eval
    # evaluation config
    parser.add_argument("--mask_mode", type=str, default='default', choices=['all_none', 'all_empty', 'empty_when_missing', 'default']) # default: read from eval data, none when missing
    parser.add_argument("--ref_mode",  type=str, default='default', choices=['all_empty', 'copy_query', 'default']) # default: read from eval data
    parser.add_argument("--ref_mask_mode",  type=str, default='default', choices=['none_for_ref', 'empty_for_ref', 'copy_query_mask', 'default', 'follow_mask_mode']) # default: use mask from eval data, if none copy query mask 
    parser.add_argument("--qa_mode",  type=str, default='default', choices=['single_qa', 'default']) # default: multi_qa according to eval data
    parser.add_argument("--image_num", type=int, default=None) # manaually set image num, will auto pad <image> and empty picture
    parser.add_argument("--overwrite_image_aspect_ratio", type=str, default=None)
    parser.add_argument("--use_pad_when_db", action='store_true') # force image_aspect_ratio use pad when input images > 1
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    
    
    
    # read data
    ds_name=args.ds
    data_dir=args.data_path
    text_src=args.text_src
    start=args.start
    if data_dir.endswith("/") and not data_dir.endswith(ds_name+"/"): # guess it is on ./evaluate/
        data_dir=os.path.join(data_dir,ds_name)
    question_path=os.path.join(data_dir,args.question_file)
    answer_path=os.path.join(data_dir,args.answer_file)
    
    all_datas=data_read(question_path)
    
    qs_tmp=None
    if text_src.startswith("config:"):
        v=text_src.replace("config:","",1)
        assert ds_name in SUPPORTED_QSET, f"{args.ds} not in eval config"
        assert v in SUPPORTED_QSET[ds_name], f"{args.ds} has no prompt version {v} in eval config"
        qs_tmp=SUPPORTED_QSET[ds_name][v]
        print(f"using prompt {v} for {ds_name} in eval config")
    elif text_src.startswith("input:"):
        qs_tmp=text_src.replace("input:","",1)
        print(f"using input prompt {qs_tmp} as question")
    elif text_src.startswith("dataset"):
        print("using prompt from eval data")
    else:
        raise ValueError(f"unkown text_src: {text_src}")
    
    if args.subset!="all":
        subsets=args.subset.split(",")
    else:
        subsets=None
    
    # load model
    worker=None
    if args.model_type=="myriad2":
        worker=SimpleModelWrapper(
            args.model_path, 
            args.model_base, 
            args.model_name, 
            'cuda',
            args.use_flash_attn,
            image_aspect_ratio=args.overwrite_image_aspect_ratio
        )
    elif args.model_type=="transformers":
        worker=TransformersModelWrapper(
            args.model_path,
            device='cuda',
        )
    elif args.model_type=="sglang":
        #worker=SGLangModelWrapper
        pass
    else:
        raise ValueError(f"unkown model_type: {args.model_type}")
    
    all_ans=[]
    # main cycle
    for d in tqdm(all_datas[start:]):
        
        subset=d["subset"]
        if subsets is not None and subset not in subsets:
            continue
            
        if type(qs_tmp) is dict:
            if subset not in qs_tmp:
                if "default" not in qs_tmp:
                    raise ValueError(f"subset {subset} in {ds_name} version {v} not in eval config")
                qs=qs_tmp["default"]
            qs=qs_tmp[subset]
        else:
            qs=qs_tmp # if qs is None, use conversation as qs
        
        # image process
        if "src" in d:
            img_p=d["src"]
        elif "image" in d and d["image"] is not None:
            img_p=d["image"]
            if type(img_p) is str:
                img_p=[img_p]
        else:
            img_p=[]
            
        imgs=[Image.open(os.path.join(data_dir,i)).convert('RGB') for i in img_p]
        
        # image num correct
        if args.image_num is None:
            img_num=len(imgs)
        else:
            img_num=args.image_num
            
        if img_num<len(imgs):
            print(f"WARNING: image_num {img_num} is smaller than len(images) {len(imgs)}, will chunk")
            imgs=imgs[:img_num]
        elif img_num>len(imgs):
            print(f"WARNING: image_num {img_num} is larger than len(images) {len(imgs)}, will pad")
            img_e=Image.new("RGB",imgs[0].size,(0,0,0))
            imgs=imgs+[img_e]*(img_num-len(imgs))
        
        assert img_num==len(imgs)
            
        
        # ref process
        img_q=imgs[0]
        img_r=imgs[1:]
        if args.ref_mode=="all_empty":
            img_r=[Image.new("RGB",i.size,(0,0,0)) for i in img_r]
        elif args.ref_mode=="copy_query":
            img_r=[img_q]*(img_num-1)
        elif args.ref_mode=="default":
            pass
        else:
            raise ValueError(f"unkown ref_mode {args.ref_mode}")
        imgs=[img_q]+img_r
        
        assert img_num==len(imgs), "after ref process"
            
        # mask process
        if "mask" in d and d["mask"] is not None:
            mask_p=d["mask"]
            if type(mask_p) is str:
                mask_p=[mask_p]
            masks=[read_mask(os.path.join(data_dir,i)) for i in mask_p]
        else:
            masks=None
        # mask num correct
        mask_o=masks
        if masks is None:
            mask_o=[]
        mask_numr=len(imgs)-len(mask_o)
        if mask_numr>0:
            mask_o=mask_o+[None]*(mask_numr)
        elif mask_numr<0:
            mask_o=mask_o[:mask_numr]
            
        # mask control
        if args.mask_mode=="all_none":
            masks=None
        elif args.mask_mode=="all_empty":
            masks=[]
            for img in imgs:
                img_np=img.numpy()
                masks.append(np.zeros(img_np.shape[:2],dtype=img_np.dtype))
        elif args.mask_mode=='empty_when_missing':
            masks=[]
            for idx,img in enumerate(imgs):
                if mask_o[idx] is None:
                    img_np=img.numpy()
                    masks.append(np.zeros(img_np.shape[:2],dtype=img_np.dtype))
                else:
                    masks.append(mask_o[idx])
        elif args.mask_mode=='default':
            masks=mask_o # no process, none when missing
        else:
            raise ValueError(f"unkown mask_mode {args.mask_mode}")
        
        # ref mask control
        mask_q=masks[0]
        mask_r=masks[1:]
        mask_rnum=len(mask_r)
        if args.ref_mask_mode=="none_for_ref":
            mask_r=[None]*mask_rnum
        elif args.ref_mask_mode=='empty_for_ref':
            mask_r=[]
            for img in imgs[1:]:
                img_np=img.numpy()
                mask_r.append(np.zeros(img_np.shape[:2],dtype=img_np.dtype))
        elif args.ref_mask_mode=='copy_query_mask':
            mask_r=[mask_q]*mask_rnum
        elif args.ref_mask_mode== 'default': # use eval_data, copy query if missing
            mask_r=[]
            for idx,img in enumerate(imgs[1:],start=1):
                if mask_o[idx] is None:
                    mask_r.append(mask_q)
                else:
                    mask_r.append(mask_o[idx])
        elif args.ref_mask_mode== 'follow_mask_mode':
            pass # no process
        else:
            raise ValueError(f"unkown ref_mask_mode {args.ref_mask_mode}")
        masks=[mask_q]+mask_r
        
        # bbox process
        if "bbox" in d and d["bbox"] is not None: # no process
            boxes_list=d["bbox"]
        else:
            boxes_list=None
        
        # default inputs params
        inputs={"text": qs,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_new_tokens": args.max_new_tokens,
                "use_pad_when_db": args.use_pad_when_db, 
                "images": imgs,
                "masks": masks,
                "boxes_list": boxes_list,
        }
        
        # qa control
        assert worker is not None
        ans=deepcopy(d)        
        if qs is None:
            qs=d["conversations"]
        if type(qs) is list: # assert qs is convs
            if args.qa_mode=="single_qa":
                convs=worker.multi_qa(inputs,qs,forget_state=True) # run model!!!!
            elif args.qa_mode=="default": #default
                convs=worker.multi_qa(inputs,qs) # run model!!!!
            else:
                raise ValueError(f"unkown qa_mode {args.qa_mode}") 
            ans["conversations"]=convs
        elif type(qs) is str:
            _,out=worker.single_qa(inputs) # run model!!!!
            ans["conversations"]=make_convs([qs,out])
        else:
            raise ValueError(f"unkown qs type {type(qs)}")
        
        # write ans
        data_write(ans,answer_path,"jsonl","a") # use jsonl as temp storage
        all_ans.append(ans)
        
    # main cycle finished
    data_write(all_ans,answer_path,"json","w")
    jsonl_tmp=answer_path.replace(".json",".jsonl")
    if os.path.isfile(jsonl_tmp):
        os.remove(jsonl_tmp)
