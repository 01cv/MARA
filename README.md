# Black-box-Face-Reconstruction
TOWARDS QUERY EFFICIENT AND GENERALIZABLE BLACK-BOX FACE RECONSTRUCTION ATTACK  
To be presented in 2023 International Conference on Image Processing (ICIP)

Minimum Assumption Reconstruction Attacks: Rise of Security and Privacy Threats against Face Recognition
To be presented in The 6th Chinese Conference on Pattern Recognition and Computer Vision, PRCV 2023

## Requirements (not strict)
- PyTorch 1.9.1
- Torchvision 0.10.1
- CUDA 10.1/10.2
- NOTE: Any version is ok if you can use the StyleGAN2 from the <a href="https://github.com/rosinality/stylegan2-pytorch"> stylegan2-pytorch</a> repository.

## Setup
Download pretrained encoders and StyleGAN2 weights:
- <a href="https://drive.google.com/file/d/1eVq2hhjHiO494qkDcGhG5EdxYOilu--7/view?usp=share_link">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1pDOX9_bQAgSkJp8W-EVq4iKBg07gTQLE/view?usp=drivesdk">ResNet-50</a>
- <a href="https://drive.google.com/file/d/1BDDpjhUYCwQde6KzR2ztGkMqgE8Nq9E2/view?usp=share_link">SwinTransformer-S</a>
- <a href="https://drive.google.com/file/d/1W4ZmSxm3gROz205JoikqVeHRroM2_fXY/view?usp=share_link">StyleGAN2-FFHQ-256x256</a>

Download LFW and CFP-FP datasets:
- <a href="https://drive.google.com/file/d/1lckCEDPjOFAyJRjpdWnfseqI50_yEXAW/view?usp=share_link">LFW</a>
- <a href="https://drive.google.com/file/d/1s769SGpacLQ3qDx413RVtRbYQrJfu0M3/view?usp=share_link">CFP-FP</a>

The images for LFW and CFP-FP datasets are already cropped and aligned using two different schemes: <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a> and <a href="https://github.com/JDAI-CV/FaceX-Zoo/issues/30"> FaceX-Zoo</a>.

After downloading, change the paths in ```dataset/dataset_conf.yaml``` and ```weight``` in ```encoder/encoder_conf.yaml``` accordingly.

## Usage
After the setup is done, simply run ```python reconstruct.py```.

## ICIP paper source code

[ICIP Source Code](https://github.com/1ho0jin1/Black-box-Face-Reconstruction)

## Citation
If you find this project useful in your research, please consider citing:
```
@article{dong2023reconstruct,
  title={Reconstruct face from features based on genetic algorithm using GAN generator as a distribution constraint},
  author={Dong, Xingbo and Miao, Zhihui and Ma, Lan and Shen, Jiajun and Jin, Zhe and Guo, Zhenhua and Teoh, Andrew Beng Jin},
  journal={Computers \& Security},
  volume={125},
  pages={103026},
  year={2023},
  publisher={Elsevier}
}
@inproceedings{dong2021towards,
  title={Towards generating high definition face images from deep templates},
  author={Dong, Xingbo and Jin, Zhe and Guo, Zhenhua and Teoh, Andrew Beng Jin},
  booktitle={2021 International Conference of the Biometrics Special Interest Group (BIOSIG)},
  pages={1--11},
  year={2021},
  organization={IEEE}
}

@inproceedings{dong2023towards,
  title={TOWARDS QUERY EFFICIENT AND GENERALIZABLE BLACK-BOX FACE RECONSTRUCTION ATTACK  },
  author={Hojin Park, Jaewoo Park, Yonsei University, Korea, Republic of; Xingbo Dong, Anhui University, China; Andrew Beng Jin Teoh, Yonsei University, Korea, Republic of},
  booktitle={2021 International Conference of the Biometrics Special Interest Group (BIOSIG)},
  pages={1--11},
  year={2021},
  organization={IEEE}
}

@inproceedings{dong2023towards,
  title={Minimum Assumption Reconstruction Attacks: Rise of Security and Privacy Threats against Face Recognition  },
  author={Dezhi Li, Hojin Park, Xingbo Dong,Yenlung Lai, Hui Zhang,Andrew Beng Jin Teoh, and Zhe Jin},
  booktitle={The 6th Chinese Conference on Pattern Recognition and Computer Vision, PRCV 2023},
  pages={1--11},
  year={2023},
  organization={Springer}
}



```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

