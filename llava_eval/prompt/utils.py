import json

def inject_json(ori_dict,json_p):
    with open(json_p) as f:
        new_dict=json.load(f)
    for k in new_dict:
       if k not in ori_dict:
           ori_dict[k]=new_dict[k]
       else:
           ori_dict[k].update(new_dict[k])
    return ori_dict
    
OTHER_DATASETS={}
OTHER_DATASETS=inject_json(OTHER_DATASETS,"llava_eval/prompt/production_wfdd.json")
OTHER_DATASETS=inject_json(OTHER_DATASETS,"llava_eval/prompt/production_pcb_bank.json")