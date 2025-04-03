import numpy as np
import jsonlines
import sys

def binary_collector(pred,label,is_pos=None,classnames="all",crt=False,eps=-1e-15,only_mean=False):
    '''
    binary_collector -> str: used to collect acc for classes, actually same function with sklearn.metrics.accuracy_score for less encapsulation,
    results will return in format can be directly store in .csv
    pred -> list[Any]: result predict by model
    label -> list[Any]: the ground truth of eval data, assert same type and length as pred, only take pred == label as true result
    is_pos -> list[Any]: the biniary label of whether the sample is postive, used to calculate recall and TNR, wrong or None will lead to non-sense recall and TNR
    classnames -> list[str]: class name for every sample, assert same length as pred, used to group result accroding to class
    crt -> bool: whether enable crt, if enabled, will assert pred -> list[list[Any]] <- label, each sample will caclulate as true only all sub pred == sub label
    eps -> float: a eps used for division in caculating mean acc
    only_mean -> bool: return all class results or just return mean result of classes
    '''
    pos_map={"True":True,"False":False,"OK":False,"good":False,"0":False,"1":True}
    outputs=[]
    all_res={}
    assert len(pred)==len(label)
    data_num=len(pred)
    if is_pos is None:
        is_pos=[True]*data_num
    elif type(is_pos) is not list:
        is_pos=[is_pos]*data_num
    if type(classnames) is not list:
        classnames=[classnames]*data_num
    for p,l,cls_n,pos in zip(pred,label,classnames,is_pos):
        if type(pos) is str and pos in pos_map:
            pos=pos_map[pos]
        res=None
        if isinstance(p,list) and not crt:
            res=(p[0]==l[0])
        elif isinstance(p,list) and crt:
            tfg=True
            nfg=False
            for j in range(len(p)):
                if p[j]!=l[j]:
                    tfg=False
                if p[j]==l[j]:
                    nfg=True
            if tfg and nfg:
                res=True
            elif not tfg and not nfg:
                res=False    
        else:
            res=(p==l)  #res: is pred consistent with label?
        if cls_n not in all_res:
            all_res[cls_n]=[0,0,0,0,0]
        if res is None:
            all_res[cls_n][4]+=1
        elif pos and res:
            all_res[cls_n][0]+=1
        elif pos and not res:
            all_res[cls_n][3]+=1
        elif not pos and not res:
            all_res[cls_n][1]+=1
        elif not pos and res:
            all_res[cls_n][2]+=1
    atp,afp,atn,afn,aun=0,0,0,0,0
    all_acc=[]
    all_pr=[]
    all_re=[]
    all_nre=[]
    outputs.append(f",acc,pre,rec,tnr,un/tot")
    dict_sort=lambda d:{i:j for i,j in sorted(d.items(),key=lambda x:x[0])}
    sn=lambda x: -1 if x<0 else x
    all_res=dict_sort(all_res)
    for i in all_res:
        tp,fp,tn,fn,un=all_res[i]
        acc=(tp+tn)/(tp+fp+tn+fn+un+eps)
        pr=tp/(tp+fp+eps)
        re=tp/(tp+fn+eps)
        nre=tn/(tn+fp+eps)
        all_acc.append(acc)
        all_pr.append(pr)
        all_re.append(re)
        all_nre.append(nre)
        outputs.append(f"{i},{sn(acc)},{sn(pr)},{sn(re)},{sn(nre)},{un}/{tp+fp+tn+fn+un}")
        atp+=tp
        afp+=fp
        atn+=tn
        afn+=fn
        aun+=un
    acc=(atp+atn)/(atp+afp+atn+afn+aun+eps)
    pr=atp/(atp+afp+eps)
    re=atp/(atp+afn+eps)
    nre=atn/(atn+afp+eps)
    mean=lambda x: sum(x)/len(x)
    mean_str=f"mean,{sn(mean(all_acc))},{sn(mean(all_pr))},{sn(mean(all_re))},{sn(mean(all_nre))},{aun}/{atp+afp+atn+afn+aun}"
    if only_mean:
        return mean_str
    outputs.append(mean_str)
    outputs.append(f"all,{sn(acc)},{sn(pr)},{sn(re)},{sn(nre)},{aun}/{atp+afp+atn+afn+aun}")
    return "\n".join(outputs)

def save_csv(f_path,fstr):
    with open(f_path,"w") as f:
        f.write(fstr+"\n")

def keyword_resolve(old_value):
    '''
    resolve keyword logic string
    '''
    l=old_value.strip().split("+")
    res=[]
    for s in l:
        r=s.split("*")
        res.append(r)
    return res

def keyword_judge(datas):
    '''
    extract pred by keyword mapping from llava train data format
    '''
    pred=[]
    for data in datas:
        conv=data["conversations"]
        for c in conv:
            if c["from"]=="gpt" and "old_value" in c:
                v=c["value"].lower()
                o=c["old_value"].lower()
                keywords=keyword_resolve(o)
                kflag=False
                for klist in keywords:
                    subflag=True
                    for k in klist:
                        if k not in v:
                            subflag=False
                            break
                    if subflag:
                        kflag=True
                        break
                pred.append(kflag)
    return pred,[True]*len(pred)

def extract_pred_label(datas,only_last=False):
    '''
    extract pred and label str from ans in llava train data format, will convert all string to lower case for fully matching
    datas -> list[dict]: all datas need to extract in llava train data format
    only_last -> bool: if True, only the last conversation in ans data is count.
    '''
    pred=[]
    label=[]
    for data in datas:
        conv=data["conversations"]
        if only_last:
            pred.append(conv[-1]["value"].lower())
            label.append(conv[-1]["old_value"].lower())
        else:
            for c in conv:
                if c["from"]=="gpt" and "old_value" in c:
                    pred.append(c["value"].lower())
                    label.append(c["old_value"].lower())
    return pred,label

def extract_pred_label_choice(datas,label_key="gt",key_map={"1":"A","0":"B"},only_last=True):
    '''
    extract pred and label from ans in llava train data format, will load biniary gt from ans file
    datas -> list[dict]: all datas need to extract in llava train data format
    label_key -> str: the name of label in ans data, default is "gt", will assert label in ans data
    key_map -> str: map "gt" value to same format with pred
    only_last -> bool: if True, only the last conversation in ans data is count, for this function, only_last should always to be true.
    '''
    pred=[]
    label=[]
    for data in datas:
        conv=data["conversations"]
        if only_last: # choice only extract the last one
            pred.append(conv[-1]["value"].strip().strip(",.").strip())
            l=data.get(label_key,None)
            if l is None:
                raise ValueError(f"data item {data} has no keyword {label_key} for label")
            if l not in key_map:
                if type(l) is int:
                    l=str(l)
            assert l in key_map, f"label {l} is not in key_map {key_map} for data item {data}"
            l=key_map[l]
            label.append(l)
    return pred,label
