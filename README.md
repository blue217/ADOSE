# Introduction

This repository is the formal implement of our paper titled “[Robust Domain Misinformation Detection via Multi-modal Feature Alignment](https://ieeexplore.ieee.org/abstract/document/10288548/)”. The contribution of this work can be summarized as follows:

1. A unified framework that tackles the domain generalization (target domain data is unavailable) and domain adaptation tasks (target domain data is available). This is necessary as obtaining sufficient unlabeled data in the target domain at an early stage of misinformation dissemination is difficult.
2. Inter-domain and cross-modality alignment modules that reduce the domain shift and the modality gap. These modules aim at learning rich features that allow misinformation detection. Both modules are plug-and-play and have the potential to be applied to other multi-modal tasks.

Additionally, we believe that the multimodal generalization algorithms proposed in our work can be used in other multimodal tasks. If you have some questions related to this paper, please feel no hesitate to ask me. 

# To run our code

You can run our codes as below:

   ```
   sh train.sh
   ```

# Citation

If you find this repository helpful, please cite our paper:

```
@ARTICLE{10288548,
  author={Liu, Hui and Wang, Wenya and Sun, Hao and Rocha, Anderson and Li, Haoliang},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Robust Domain Misinformation Detection via Multi-modal Feature Alignment}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIFS.2023.3326368}}
```

If you have interest in multimodal misinformation detection, another paper of me on multimodal misinformation task can help you https://arxiv.org/abs/2305.05964. 

```
@inproceedings{DBLP:conf/acl/LiuWL23,
  author       = {Hui Liu and
                  Wenya Wang and
                  Haoliang Li},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Interpretable Multimodal Misinformation Detection with Logic Reasoning},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {9781--9796},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.620},
  doi          = {10.18653/V1/2023.FINDINGS-ACL.620},
  timestamp    = {Thu, 10 Aug 2023 12:35:42 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/LiuWL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
