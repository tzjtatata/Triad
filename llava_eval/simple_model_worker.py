import json
import time
import math
import os
import torch
from tqdm import tqdm
import requests
import shortuuid
import hashlib
import numpy as np

# from evaluation_config import *

from PIL import Image

from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)


class SimpleModelWorker:
    def __init__(self, model_path, model_base, model_name, device, 
                 load_8bit=False, load_4bit=False, use_flash_attn=False, image_aspect_ratio=None):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        print(f"Loading the model {self.model_name} ...")
        # @lyz(2025-01-03)
        # vocab_size = 152064 if '0.5b' not in self.model_name else 151936
        # ow_cfg={"vocab_size":vocab_size,}
        ow_cfg = {}
        if image_aspect_ratio is not None:
            print(f"overwrite image_aspect_ratio {image_aspect_ratio}")
            ow_cfg["image_aspect_ratio"]=image_aspect_ratio
            if image_aspect_ratio=="randomroi":
                ow_cfg["mm_patch_merge_type"]="spatial_avgpool_auto_unpad_add_newl"
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device, use_flash_attn=use_flash_attn,
            overwrite_config=ow_cfg)
        self.is_multimodal = 'llava' in self.model_name.lower()

    @torch.inference_mode()
    def generate_batch(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        masks = params.get("masks", None)
        boxes_list = params.get("boxes_list", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                # images = [load_image_from_base64(image) for image in images]
                image_sizes = [image.size for image in images]
                # 这里不做masks和boxes的选择，在process_images里做。
                if len(images)>1 and params.get('use_pad_when_db', False):
                    images = process_images(images, image_processor, model.config, masks=masks, boxes_list=boxes_list, overwrite_img_asp="pad")
                else:
                    images = process_images(images, image_processor, model.config, masks=masks, boxes_list=boxes_list)

                if type(images) is list:
                    if type(images[0]) is not dict:
                        images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
                image_sizes = None
            image_args = {"images": images, "image_sizes": image_sizes}
        else:
            images = None
            image_args = {}
        if '<image>' in prompt: assert images is not None
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            print(ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.")
            return ""

        
        data = model.generate(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **image_args
        )
        return ''.join(tokenizer.batch_decode(
            data, 
            skip_special_tokens=True
        ))

class SimpleModelWrapper:
    def __init__(self,*args,**kwargs):
        self.model=SimpleModelWorker(*args,**kwargs)
        self.model_name=self.model.model_name
        self.template_name=self.select_template(self.model_name)
        print("using conversation template",self.template_name)
        
    def select_template(self, model_name):
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "qwen" in model_name.lower():
                template_name = "qwen_1_5"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            # elif 'llava-v1.6-vicuna-13b' in model_name.lower():
            #     template_name = "vicuna_v1"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        return template_name
    
    def add_text(self, state, text, image, image_process_mode="Default"):  # Follow gradio setting, the image_process_mode Radio is not visible and not changable.
        assert len(text) <= 8192, "测试的text超过了8192个字符."
        text = text[:8192]  # Hard cut-off # origin 1536
        if image is not None:
            assert len(text) <= 4096, "多模态时测试的text超过了4096个字符." 
            text = text[:4096]  # Hard cut-off for images # origin 1200
            if '<image>' not in text:
                # text = '<Image><image></Image>' + text
                # text = text + '\n<image>'
                text = '<image>\n' + text
            text = (text, image, image_process_mode)
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False
        return state
    
    def new_state(self):
        return conv_templates[self.template_name].copy()
    
    def single_qa(self,inputs,state=None,multi_round=False):
        if state is None:
            state=self.new_state()
        image_num=0
        imgs=inputs.get("images",None)
        if imgs is not None:
            image_num=len(imgs)
        qs=inputs.get("text",None)
        state=self.add_text(state, qs, None if multi_round else imgs)
        prompt=state.get_prompt()
        if prompt.count("<image>")==1 and prompt.count("<image>") != image_num:
            img_s="\n".join(["<image>"]*image_num)
            prompt=prompt.replace("<image>",img_s)
        out=self.model.generate_batch({
                "model": self.model_name,
                "prompt": prompt,
                "temperature": inputs.get("temperature",0.2),
                "top_p":  inputs.get("top_p",0.7),
                "max_new_tokens": inputs.get("max_new_tokens",512),
                "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
                "use_pad_when_db": inputs.get("use_pad_when_db",False), 
                "images": imgs,
                "masks": inputs.get("masks",None),
                "boxes_list": inputs.get("boxes_list",None),
            }).strip()
        state.messages[-1][-1]=out
        return state,out
    
    # 封装了多轮对话
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self.new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            if forget_state:
                state=self.new_state()
            state,out=self.single_qa(inputs,state,conv_idx!=0)
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs
        

class TransformersModelWrapper:
    """
    Refer to [
        https://qwen.readthedocs.io/en/latest/inference/chat.html.  (LLM)
        https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct  (Qwen2VL)
        https://huggingface.co/docs/transformers/main/en/model_doc/llava_onevision#single-image-inference  (LLAVA-OneVision)
        https://huggingface.co/openbmb/MiniCPM-V-2_6  (MiniCPM-V 2.6)
    ]
    """
    def __init__(self, model_path, task='LMM', **kwargs):
        import torch
        self.task = task
        assert task in ['LLM', 'LMM'], f"TransformersModelWrapper不支持{task}类型."
        self.device = kwargs.get('device', None)
        if self.task == 'LLM':
            from transformers import pipeline
            
            if self.device is not None:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device=self.device)

            else:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device_map="auto")

        if self.task == 'LMM':
            from transformers import AutoTokenizer, AutoProcessor, AutoModel
            from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration

            if self.device is not None:
                device_map = self.device
            else:
                device_map = "auto"

            if "Qwen2-VL" in model_path:
                self.model_type = 'qwen2'
                model_class = Qwen2VLForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, torch_dtype="auto", device_map=device_map
                )
                
            elif 'lmms-lab--llava-onevision' in model_path and '-hf' in model_path:  # 这种方式只能调用hf格式的llava-onevision模型。
                self.model_type = 'llava-onevision'
                model_class = LlavaOnevisionForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, torch_dtype="auto", device_map=device_map
                )
            
            elif 'MiniCPM-V' in model_path:
                self.model_type = 'minicpm-v'
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True,
                    attn_implementation='sdpa', torch_dtype=torch.bfloat16
                )
                self.model.eval()
                if self.device is not None:
                    self.model.to(self.device)
                
            else:
                raise NotImplementError(f"不支持多模态模型: {model_path}")
            
            self.processor = AutoProcessor.from_pretrained(model_path)
    
    def add_text(self, state, text, new_images=None, **kwargs):
        if self.task == 'LLM':
            assert images is None, "语言模型不支持输入图像."
            state.append({"role": "user", "content": text})
            return state
        elif self.task == 'LMM':
            if new_images is not None:
                assert isinstance(new_images, list), "输入的图像应该是一个队列List."
                if self.model_type != 'minicpm-v':
                    state['state'].append(
                        {
                            "role": "user",
                            "content": [
                                    {
                                        "type": "image",
                                    }
                                    for _ in range(len(new_images))
                                ]
                                +
                                [
                                    {"type": "text", "text": text},
                                ],
                        }
                    )
                    state['images'] += new_images
                else:
                    state['state'].append(
                        {
                            "role": "user",
                            "content": new_images + [text]
                        }
                    )
                
            else:
                state['state'].append({"role": "user", "content": text})
            return state
        else:
            raise NotImplementError(f"不支持{self.task}的任务类型")
    
    def new_state(self):
        if self.task == 'LLM':
            return []
        elif self.task == 'LMM':
            return {
                "state": [], 
                "images": []
            }
        else:
            raise NotImplementError(f"不支持{self.task}的任务类型")

    def generate(self, new_text, images=None, state=None,  **generate_kwargs):
        if state is None:
            state = self.new_state()
        
        state = self.add_text(state, new_text, new_images=images)
        if self.task == 'LLM':
            response_message = self.model(state, **generate_kwargs)[0]["generated_text"][-1]
            state.append(response_message)
            return state, response_message['content']
        elif self.task == 'LMM':
            if self.model_type != 'minicpm-v':
                text_prompt = self.processor.apply_chat_template(state['state'], add_generation_prompt=True)
                # print(text_prompt)
                if self.device is None:  # 72B这里写的也是CUDA……有可能是device_map=auto可以这么弄
                    inputs = self.processor(
                        text=[text_prompt], images=state['images'], padding=True, return_tensors="pt"
                    ).to('cuda')
                else:
                    inputs = self.processor(
                        text=[text_prompt], images=state['images'], padding=True, return_tensors="pt"
                    ).to(self.device)

                output_ids = self.model.generate(**inputs, **generate_kwargs)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': output_text[0],
                })
                return state, output_text[0]
            else:
                answer = self.model.chat(
                    image=None,
                    msgs=state['state'], 
                    tokenizer=self.tokenizer
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': [answer]
                })
                # print(answer)
                return state, answer

        else:
            raise NotImplementError(f"不支持{self.task}的任务类型")
    
    def single_qa(self, inputs, state=None):
        if state is None:
            images=inputs.get("images",None)
            assert images is not None and len(images) == 1, "0-shot时必须有一张待测图像作为输入. "

        qs=inputs.get("text",None)

        generate_kwargs = {
            "temperature": inputs.get("temperature",0.2),
            "top_p":  inputs.get("top_p",0.7),
            "max_new_tokens": inputs.get("max_new_tokens",512),
        }

        return self.generate(qs, state=state, images=images, **generate_kwargs)
    
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self.new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            if forget_state:
                state=self.new_state()
            state,out=self.single_qa(inputs,state,conv_idx!=0)
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs
