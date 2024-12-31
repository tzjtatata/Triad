#  Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process.

**Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process** [[Paper]()] [[HF](coming soon)] <br>
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
Coming soon.

## Triad Weights
Triad Weights are coming soon. 

## Demo

The demo code is coming soon.

## Train

TODO. 

## Evaluation

In Traid, we evaluate models on the public benchmark for Anomaly Detection, MVTec, WFDD and PCB Bank. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search.

## Citation

If you find Triad useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{Myriad,
  title={Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process},
  author={Li, Yuanze and Yuan, Shihao and Wang, Haolin and Li, Qizhang and Liu, Ming and Xu, Chen and Shi, Guangming and Zuo, Wangmeng},
  journal={},
  year={2025}
}
```

## Acknowledgement

- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT): the codebase we built upon. Thanks for their clear code base for reproduce, finetune and DPO training!
- [LLaVA-1.6](https://github.com/haotian-liu/LLaVA): we also employ Triad on LLaVA-1.6. Thanks for their code base and AnyRes module which is inspired us to build the proposed EG-RoI module for IAD domain!
