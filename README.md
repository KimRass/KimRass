# 1. Personal Projects

## 1) From-scratch PyTorch Implementations of AI papers
|분류 및 연도|논문|내용|
|:-:|:-|:-|
|Vision|
|2014|[VAE](https://github.com/KimRass/VAE)<br>Kingma and Welling|[✓] Training on MNIST<br>[✓] Visualizing Encoder output<br>[✓] Visualizing Decoder output<br>[✓] Reconstructing image|
|2015|[CAM](https://github.com/KimRass/CAM)<br>Zhou et al.|[✓] Applying GoogLeNet<br>[✓] Generating 'Class Activatio Map'<br>[✓] Generating bounding box|
|2016|[Gatys et al.](https://github.com/KimRass/Gatys-et-al.-2016)|[✓] Experimenting on input image size<br>[✓] Experimenting on VGGNet-19 with Batch normalization<br>[✓] Applying VGGNet-19|
||[YOLO](https://github.com/KimRass/YOLO)<br>Redmon et al.|[✓] Model architecture<br>[✓] Visualizing ground truth on grid<br>[✓] Visualizing model output<br>[✓] Visualizing class probability map<br>[ㅤ] Loss function<br>[ㅤ] Training on VOC 2012|
||[DCGAN](https://github.com/KimRass/DCGAN)<br>Radford et al.|[✓] Training on CelebA at 64 × 64<br> [✓] Sampling<br>[✓] Interpolating in latent space<br>[ㅤ] Training on CelebA at 32 × 32|
||[Noroozi et al.](https://github.com/KimRass/Mehdi-Noroozi-et-al.-2016)|[✓] Model architecture<br>[✓] Chromatic aberration<br>[✓] Permutation set|
||[Zhang et al.](https://github.com/KimRass/Richard-Zhang-et-al.-2016)|[✓]  Visualizing empirical probability distribution<br>[ㅤ] Model architecture<br>[ㅤ] Loss function<br>[ㅤ] Training|
|2014<br>2017|[Conditional GAN](https://github.com/KimRass/Conditional-WGAN-GP)<br>Mirza et al.<br>[WGAN-GP](https://github.com/KimRass/Conditional-WGAN-GP)<br>Gulrajani et al.|[✓] Training on MNIST|
|2016<br>2017|[VQ-VAE](https://github.com/KimRass/VQ-VAE-PixelCNN)<br>Oord et al.<br>[PixelCNN](https://github.com/KimRass/VQ-VAE-PixelCNN)<br>Oord et al.|[✓] Training on Fashion MNIST<br>[✓] Training on CIFAR-10<br>[✓] Sampling|
|2017|[Pix2Pix](https://github.com/KimRass/Pix2Pix)<br>Isola et al.|[✓] Experimenting on image mean and std<br>[✓] Experimenting on `nn.InstanceNorm2d()`<br>[✓] Training on Google Maps<br>[✓] Training on Facades<br>[ㅤ] higher resolution input image|
||[CycleGAN](https://github.com/KimRass/CycleGAN)<br>Zhu et al.|[✓] Experimenting on random image pairing<br>[✓] Experimenting on LSGANs<br>[✓] Training on monet2photo<br>[✓] Training on vangogh2photo<br>[✓] Training on cezanne2photo<br>[✓] Training on ukiyoe2photo<br>[✓] Training on horse2zebra<br>[✓] Training on summer2winter_yosemite|
|2018|[PGGAN](https://github.com/KimRass/PGGAN)<br>Karras et al.|[✓] Experimenting on image mean and std<br>[✓] Training on CelebA-HQ at 512 × 512<br>[✓] Sampling|
||[DeepLabv3](https://github.com/KimRass/DeepLabv3)<br>Chen et al.|[✓] Training on VOC 2012<br>[✓] Predicting on VOC 2012 validation set<br>[✓] Average mIoU<br>[✓] Visualizing model output|
||[RotNet](https://github.com/KimRass/RotNet)<br>Gidaris et al|[✓] Visualizing Attention map|
||[StarGAN](https://github.com/KimRass/StarGAN)<br>Yunjey Choi et al.|[✓] Model architecture|
|2020|[STEFANN](https://github.com/KimRass/STEFANN)<br>Roy et al.|[✓] FANnet architecture<br>[✓] Colornet architecture<br>[✓] Training FANnet on Google Fonts<br>[✓] Custom Google Fonts dataset<br>[✓] Average SSIM<br>[ㅤ] Training Colornet|
||[DDPM](https://github.com/KimRass/DDPM)<br>Ho et al.|[✓] Training on CelebA at 32 × 32<br>[✓] Training on CelebA at 64 × 64<br>[✓] Visualizing denoising process<br>[✓] Sampling using linear interpolation<br>[✓] Sampling using coarse-to-fine interpolation|
||[DDIM](https://github.com/KimRass/DDIM)<br>Song et al.|[✓] Normal sampling<br>[✓] Sampling using spherical linear interpolation<br>[✓] Sampling using grid interpolation<br>[✓] Truncated normal|
||[ViT](https://github.com/KimRass/ViT)<br>Dosovitskiy et al.|[✓] Training on CIFAR-10<br>[✓] Training on CIFAR-100<br>[✓] Visualizing Attention map using Attention Roll-out<br>[✓] Visualizing position embedding similarity<br>[✓] Interpolating position embedding<br>[✓] CutOut<br>[✓] CutMix<br>[✓] Hide-and-Seek|
||[SimCLR](https://github.com/KimRass/SimCLR)<br>Chen et al.|[✓] Normalized temperature-scaled cross entropy loss<br>[✓] Data augmentation<br>[✓] Pixel intensity histogram|
||[DETR](https://github.com/KimRass/DETR)<br>Carion et al.|[✓] Model architecture<br>[ㅤ] Bipartite matching & loss<br>[ㅤ] Batch normalization freezing<br>[ㅤ] Training on COCO 2017
|2021|[Improved DDPM](https://github.com/KimRass/Improved-DDPM)<br>Nichol and Dhariwal|[✓] Cosine diffusion schedule|
||[Classifier-Guidance](https://github.com/KimRass/Classifier-Guidance)<br>Dhariwal and Nichol|[✓] Training on CIFAR-10<br>[ㅤ] AdaGN<br>[ㅤ] BiGGAN Upsample/Downsample<br>[ㅤ] Improved DDPM sampling<br>[ㅤ] Conditional/Unconditional models<br>[ㅤ] Super-resolution model<br>[ㅤ] Interpolation|
||[ILVR](https://github.com/KimRass/ILVR)<br>Choi et al.|[✓] Sampling using single reference<br>[✓] Sampling using various downsampling factors<br>[✓] Sampling using various conditioning range|
||[SDEdit](https://github.com/KimRass/SDEdit)<br>Meng et al.|[✓] User input stroke simulation<br>[✓] Applying CelebA at 64 × 64|
||[MAE](https://github.com/KimRass/MAE)<br>He et al.|[✓] Model architecture for pre-training<br>[ㅤ] Model architecture for self-supervised learning<br>[ㅤ] Training on ImageNet-1K<br>[ㅤ] Fine-tuning<br>[ㅤ] Linear probing|
||[Copy-Paste](https://github.com/KimRass/Copy-Paste)<br>Ghiasi et al.|[✓] COCO dataset processing<br>[✓] Large scale jittering<br>[✓] Copy-Paste (within mini-batch)<br>[✓] Visualizing data<br>[ㅤ] Gaussian filter|
||[ViViT](https://github.com/KimRass/ViViT)<br>Arnab et al.|[✓] 'Spatio-temporal attention' architecture<br>[✓] 'Factorised encoder' architecture<br>[✓] 'Factorised self-attention' architecture|
|2022|[CFG](https://github.com/KimRass/CFG)<br>Ho et al.|
|Language|
|2017|[Transformer](https://github.com/KimRass/Transformer)<br>Vaswani et al.|[✓] Model architecture<br>[✓] Visualizing position encoding|
|2019|[BERT](https://github.com/KimRass/BERT)<br>Devlin et al.|[✓] Model architecture<br>[✓] Masked language modeling<br>[✓] BookCorpus data processing<br>[✓] SQuAD data processing<br>[✓] SWAG data processing|
||[Sentence-BERT](https://github.com/KimRass/Sentence-BERT)<br>Reimers et al.|[✓] Classification loss<br>[✓] Regression loss<br>[✓] Constrastive loss<br>[✓] STSb data processing<br>[✓] WikiSection data processing<br>[ㅤ] NLI data processing|
||[RoBERTa](https://github.com/KimRass/RoBERTa)<br>Liu et al.|[✓] BookCorpus data processing<br>[✓] Masked language modeling<br>[ㅤ] BookCorpus data processing ('SEGMENT-PAIR' + NSP)<br>[ㅤ] BookCorpus data processing ('SENTENCE-PAIR' + NSP)<br>[✓] BookCorpus data processing ('FULL-SENTENCES')<br>[ㅤ] BookCorpus data processing ('DOC-SENTENCES')|
|2021|[Swin Transformer](https://github.com/KimRass/Swin-Transformer)<br>Liu et al.|[✓] Patch partition<br>[✓] Patch merging<br>[✓] Relative position bias<br>[✓] Feature map padding<br>[✓] Self-attention in non-overlapped windows<br>[ㅤ] Shifted Window based Self-Attention|
|2024|[RoPE](https://github.com/KimRass/RoPE)<br>Su et al.|[✓] Rotary Positional Embedding|
|Vision-Language|
|2021|[CLIP](https://github.com/KimRass/CLIP)<br>Radford et al.|[✓] Training on Flickr8k + Flickr30k<br>[✓] Zero-shot classification on ImageNet1k (mini)<br>[✓] Linear classification on ImageNet1k (mini)|
<!-- ||[Noroozi et al.](https://github.com/KimRass/Mehdi-Noroozi-et-al.-2017)|[✓] Model architecture<br>[✓] Constrastive loss| -->
<!-- ||[PixelLink](https://github.com/KimRass/PixelLink)|Deng et al.|[✓] Model architecture<br>[✓] Instance-balanced cross entropy loss<br>[✓] Model output post-processing| -->

## 2) [Fine-tuning 'EasyOCR' on the '공공행정문서 OCR' Dataset Provided by 'AI-Hub'](https://github.com/KimRass/train_easyocr)

## 3) [Recognizing Book Content Using the 'CLOVA OCR API'](https://github.com/KimRass/book_text_recognizer)

## 4) [A Rule-based Algorithm for Solving Edge-matching Puzzles of Arbitrary Sizes Using L2 Distance](https://github.com/KimRass/Jigsaw-Puzzle)

## 5) [A 'FastAPI'-based API for Performing Semantic Segmentation Using a 'DeepLabv3' Pretrained on the 'VOC2012' dataset](https://github.com/KimRass/FastAPI)

<!-- # 2. Resume
- [이력서-김종범](https://github.com/KimRass/KimRass/blob/main/이력서-김종범.md) -->

<!-- # 2. Github Stats
![KimRass' GitHub stats](https://github-readme-stats.vercel.app/api?username=KimRass&show_icons=true&theme=radical) -->
<!-- 
# 2. 'Baekjoon Online Judge' Solved Rank
![hyp3rflow's solved.ac stats](https://github-readme-solvedac.hyp3rflow.vercel.app/api/?handle=rmx1000)
 -->