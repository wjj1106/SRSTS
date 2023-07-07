# SRSTS & SRSTS v2
[SRSTS](https://arxiv.org/pdf/2207.07253v2.pdf) is an effective single shot text spotter which decouples text recognition from detection. It's published in ACM Multimedia 2022. [SRSTS v2](https://arxiv.org/pdf/2207.07253.pdf) the extended version which redesigns the text detection module to enable the collaborative optimization and mutual enhancement between text detection and recogntion.

# SRSTS
## Training
To be released.

## Testing
You can download benchmarks from [BaiduNetDisk](https://pan.baidu.com/s/1Mob5nzDREu0eqzbAQZqNcA
)(code: cctq) or [Google Drive](https://drive.google.com/drive/folders/1kVLtz_prtEe3hzNC-9kTV5HH9xx67bkp) and change the root in config file.

You can download the trained models from [BaiduNetDisk](https://pan.baidu.com/s/1Bcf73wCW6VM0cirVCmCzQw) (code: hf8a) or [Google Drive](https://drive.google.com/drive/folders/1mMJwkzM7wxvaGY53HG9uhELtgg7vObdy).
### Performance Evaluation

```
python scripts/test_model.py --config-file="path to the config file" --ckpt="path to your ckpt"
```

Evalute on Total-Text:
```
python scripts/test_model.py --config-file="./configs/evaluation/v1/tt.yaml" --ckpt="./save_models/tt_best.pth" --lexicon_type=0 # 0 refers none, 1 refers to full
```

Evalute on ICDAR 2015:

```
python scripts/test_model.py --config-file="./configs/evaluation/v1/ic15.yaml" --ckpt="./save_models/ic15_best.pth"  --lexicon_type=0 #0 refers none, 1 refers generic
```
Evalute on CTW 1500:

```
python scripts/test_model.py --config-file="./configs/evaluation/v1/ctw.yaml" --ckpt="./save_models/ctw_best.pth"  --lexicon_type=0 #0 refers none, 1 refers full
```

# SRSTS v2
To be released.

# Citation
Please cite the related works in your publications if it helps your research:

```
@inproceedings{wu2022decoupling,
  title={Decoupling recognition from detection: Single shot self-reliant scene text spotter},
  author={Wu, Jingjing and Lyu, Pengyuan and Lu, Guangming and Zhang, Chengquan and Yao, Kun and Pei, Wenjie},
  booktitle={ACM MM},
  pages={1319--1328},
  year={2022}
}
```
```
@misc{wu2023single,
      title={Single Shot Self-Reliant Scene Text Spotter by Decoupled yet Collaborative Detection and Recognition}, 
      author={Jingjing Wu and Pengyuan Lyu and Guangming Lu and Chengquan Zhang and Wenjie Pei},
      year={2023},
      eprint={2207.07253},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
