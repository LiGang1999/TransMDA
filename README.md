# Official implementation for TransMDA

Official pytorch implementation for [Transformer-Based Multi-Source Domain Adaptation Without Source Data](https://doi.org/10.1109/IJCNN54540.2023.10191276).

## Prerequisites (Python 3.9.12)

```
pip install -r requirements.txt
```

## Model preparation

We choose R50-ViT-B_16 as our encoder.

```bash root transformerdepth
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz 
mkdir ./model/vit_checkpoint/imagenet21k 
mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## Training

### Office-Home (for example)

#### Source pretraining

```
CUDA_VISIBLE_DEVICES=0 python image_source.py --trte val --output ckps/source/ --da uda --dset office-home --max_epoch 100 --s 0 > log_s0_office-home.txt
```

#### Target adaptation

```
CUDA_VISIBLE_DEVICES=0 python image_target_Multi_Src.py --da uda --dset office-home --t 0 --output_src ckps/source/ --output ckps/target_Multi_Src/ --pls True > log_t0_office-home.txt
```

## Reference

If you find this useful in your work, please consider citing our paper:

```
@inproceedings{li2023transformer,
  title={Transformer-Based Multi-Source Domain Adaptation Without Source Data},
  author={Li, Gang and Wu, Chao},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgements

[TransDA](https://github.com/ygjwd12345/TransDA)

[DECISION](https://github.com/driptaRC/DECISION)

[KD3A](https://github.com/FengHZ/KD3A)
