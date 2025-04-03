import sys
import json
import jsonlines
import os
from collect_utils import binary_collector,save_csv,extract_pred_label_choice

rts=sys.argv[1:]

if __name__=="__main__":
    for rt in rts: # for every ans file to collect
        # read ans data
        fdir,fn=os.path.split(rt)
        fn,ext=os.path.splitext(fn)
        if ext==".jsonl":
            with jsonlines.open(rt) as f:
                all_datas=list(f)
        elif ext==".json":
            with open(rt) as f:
                all_datas=json.load(f)
        else:
            raise ValueError(f"unknown ext for {rt}, supposed .json or .jsonl")
        # is_pos to label if sample need to be calculated for recall and TNR
        is_pos=[d["gt"] for d in all_datas]
        # classnames for every sample used to calculate mean for class
        classnames=[d["subset"] for d in all_datas]
        # extract pred and label from ans in format of llava train data
        pred,label=extract_pred_label_choice(all_datas)
        # actually same function with sklearn.metrics.accuracy_score for less encapsulation
        res=binary_collector(pred,label,is_pos=is_pos,classnames=classnames) # pass is_pos if you want correct TNR
        print(res)
