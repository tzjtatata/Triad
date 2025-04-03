#  Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process.

**Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process** [[Paper](https://arxiv.org/abs/2503.13184)] [[HF](coming soon)] <br>
Yuanze Li*, Shihao Yuan*, Haolin Wang, Qizhang Li, Ming Liu (csmliu@outlook.com), Chen Xu, Guangming Shi, Wangmeng Zuo



## TODO
- [ ] upload Triad codes (LLaVA-OneVision Version. ).
- [ ] update Triad model pretrained weights.
- [ ] update human annotated instruction datasets for IAD: instructIAD.


## Contents
- [Install](#install)
- [Triad Weights](#Triad-Weights)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
Create a python 3.10 environment and install dependencies in `requirements.txt` using following commands.

```
conda create -n myriad2_ov python==3.10
pip install -r requirements.txt
```

You may need install `flash-attn` mananully using following command.
```
pip install flash-attn --no-build-isolation
```

## Triad Weights
Triad Weights are coming soon. 

## Demo

The demo code is coming soon.

## Train

TODO. 

## Evaluation

### Requirement

Required Sigclip pretrained model to build the vision tower. Download the weights to google/siglip-so400m-patch14-384.

Required Myriad2 pretrained model. 

### Usage

In Traid, we evaluate models on the public benchmark for Anomaly Detection, MVTec, WFDD and PCB Bank. To ensure the reproducibility, we evaluate the models with greedy decoding instead of beam search.

Download evaluation data and put them under `evaluation` directory.

Download our finetuned models or finetune model by yourself and put them under the `checkpoints` directory. Then run following command to evaluate 0shot acc on MVTec AD, WFDD and PCB Bank.

```
bash eval_0shot_path.sh ./checkpoints/YOUR_CHEKCPOINTS_PATH
```

Following command for 1shot acc on MVTec AD.

```
bash eval_1shot_path.sh ./checkpoints/YOUR_CHEKCPOINTS_PATH
```

## Citation

If you find Triad useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{Myriad2,
  title={Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process},
  author={Li, Yuanze and Yuan, Shihao and Wang, Haolin and Li, Qizhang and Liu, Ming and Xu, Chen and Shi, Guangming and Zuo, Wangmeng},
  journal={},
  year={2025}
}
```

## Acknowledgement

- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT): the codebase we built upon. Thanks for their clear code base for reproduce, finetune and DPO training!
- [LLaVA-1.6](https://github.com/haotian-liu/LLaVA): we also employ Triad on LLaVA-1.6. Thanks for their code base and AnyRes module which is inspired us to build the proposed EG-RoI module for IAD domain!
