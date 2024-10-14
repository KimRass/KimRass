# 1. From-scratch PyTorch Implementations
|분류 및 연도|이름|저자|구현 내용|
|:-:|:-|:-|:-|
|Vision|
|2014|[VAE](https://github.com/KimRass/VAE)|Kingma and Welling|[✓] Training on MNIST<br>[✓] Encoder output visualization<br>[✓] Decoder output visualization|
|2015|[CAM](https://github.com/KimRass/CAM)|Zhou et al.|[✓] Application to GoogleNet<br>[✓] Bounding box generation from Class Activation Map|
|2016|[Gatys et al., 2016 (image style transfer)](https://github.com/KimRass/Gatys-et-al.-2016)|Gatys et al.|[✓] Application to VGGNet-19|
||[YOLO](https://github.com/KimRass/YOLO)|Redmon et al.|[✗] Training on VOC 2012<br>[✗] Class probability map<br>[✗] Ground truth vlisualization on grid|
||[DCGAN](https://github.com/KimRass/DCGAN)|Radford et al.|[✓] Training on CelebA at 64 × 64<br> [✓] Sampling<br>[✓] Latent space interpolation|
||[Noroozi et al., 2016](https://github.com/KimRass/Mehdi-Noroozi-et-al.-2016)|Noroozi et al.|[✓] Model architecture<br>[✓] Chromatic aberration<br>[✓] Permutation set|
||[Zhang et al., 2016 (image colorization)](https://github.com/KimRass/Richard-Zhang-et-al.-2016)|Zhang et al.|[✓] Empirical probability distribution visualization<br>[✗] Color space|
|2014<br>2017|[Conditional GAN<br>WGAN-GP](https://github.com/KimRass/Conditional-WGAN-GP)|Mirza et al.<br>Gulrajani et al.|[✓] Training on MNIST|
|2016<br>2017|[VQ-VAE & PixelCNN](https://github.com/KimRass/VQ-VAE-PixelCNN)|Oord et al.<br>Oord et al.|[✓] Training on Fashion MNIST<br>[✓] Training on CIFAR-10|
|2017|[Pix2Pix](https://github.com/KimRass/Pix2Pix)|Isola et al.|[✓] Training on Google Maps<br>[✓] Training on Facades<br> [✗] Inference on larger resolution|
||[CycleGAN](https://github.com/KimRass/CycleGAN)|Zhu et al.|[✓] Training on monet2photo<br>[✓] Training on vangogh2photo<br>[✓] Training on cezanne2photo<br>[✓] Training on ukiyoe2photo<br>[✓] Training on horse2zebra<br>[✓] Training on summer2winter_yosemite|
||[Noroozi et al., 2017](https://github.com/KimRass/Mehdi-Noroozi-et-al.-2017)|Noroozi et al.|[✓] Constrastive loss|
|2018|[PGGAN](https://github.com/KimRass/PGGAN)|Karras et al.|[✓] Training on CelebA-HQ at 512 × 512|
||[DeepLab v3](https://github.com/KimRass/DeepLabv3)|Chen et al.|[✓] Training on VOC 2012<br>[✓] Prediction on VOC 2012 validation set<br>[✓] Average mIoU<br>[✓] Model output visualization|
||[RotNet](https://github.com/KimRass/RotNet)|Gidaris et al|[✓] Attention map visualization|
||[StarGAN](https://github.com/KimRass/StarGAN)|Yunjey Choi et al.|[✓] Model architecture|
|2020|[STEFANN](https://github.com/KimRass/STEFANN)|Roy et al.|[✓] FANnet architecture<br>[✓] Training FANnet on Google Fonts<br>[✓] Custom Google Fonts dataset<br>[✓] Average SSIM|
||[DDPM](https://github.com/KimRass/DDPM)|Ho et al.|[✓] Training on CelebA at 32 × 32<br>[✓] Training on CelebA at 64 × 64<br>[✓] Denoising process visualization<br>[✓] Sampling using linear interpolation<br>[✓] Sampling using coarse-to-fine interpolation|
||[DDIM](https://github.com/KimRass/DDIM)|Song et al.|[✓] Normal sampling<br>[✓] Sampling using spherical linear interpolation<br>[✓] Sampling using grid interpolation<br>[✓] Truncated normal|
||[ViT](https://github.com/KimRass/ViT)|Dosovitskiy et al.|[✓] Training on CIFAR-10<br>[✓] Training on CIFAR-100<br>[✓] Attention map visualization using Attention Roll-out<br>[✓] Position embedding similarity visualization<br>[✓] Position embedding interpolation<br>[✓] CutOut<br>[✓] CutMix<br>[✓] Hide-and-Seek|
||[SimCLR](https://github.com/KimRass/SimCLR)|Chen et al.|[✓] Normalized temperature-scaled cross entropy loss<br>[✓] Data augmentation<br>[✓] Pixel intensity histogram|
||[DETR](https://github.com/KimRass/DETR)|Carion et al.|[✓] Model architecture<br>[✗] Bipartite matching & loss<br>[✗] Batch normalization freezing<br>[✗] Data preparation<br>[✗] Training on COCO 2017
|2021|[Improved DDPM](https://github.com/KimRass/Improved-DDPM)|Nichol and Dhariwal|[✓] Cosine diffusion schedule|
||[Classifier-Guidance](https://github.com/KimRass/Classifier-Guidance)|Dhariwal and Nichol|[✓] Training on CIFAR-10[✗] AdaGN<br>[✗] BiGGAN Upsample/Downsample<br>[✗] Improved DDPM sampling<br>[✗] Conditional/Unconditional models<br>[✗] Super-resolution model<br>[✗] Interpolation|
||[ILVR](https://github.com/KimRass/ILVR)|Choi et al.|[✓] Sampling using single reference<br>[✓] Sampling using various downsampling factors<br>[✓] Sampling using various conditioning range|
||[SDEdit](https://github.com/KimRass/SDEdit)|Meng et al.|[✓] User input stroke simulation<br>[✓] Application to CelebA at 64 × 64|
||[MAE](https://github.com/KimRass/MAE)|He et al.|[✓] Model architecture for pre-training<br>[✗] Model architecture for self-supervised learning<br>[✗] Training on ImageNet-1K<br>[✗] Fine-tuning<br>[✗] Linear probing|
||[Copy-Paste](https://github.com/KimRass/Copy-Paste)|Ghiasi et al.|[✓] COCO dataset processing<br>[✓] Large scale jittering<br>[✓] Copy-Paste (within mini-batch)<br>[✓] Data visualization<br>[✗] Gaussian filter|
||[ViViT](https://github.com/KimRass/ViViT)|Arnab et al.|[✓] 'Spatio-temporal attention' architecture<br>[✓] 'Factorised encoder' architecture<br>[✓] 'Factorised self-attention' architecture|
|2022|[CFG](https://github.com/KimRass/CFG)|Ho et al.|
|Language|
|2017|[Transformer](https://github.com/KimRass/Transformer)|Vaswani et al.|[✓] Model architecture<br>[✓] Position encoding visualization|
|2019|[BERT](https://github.com/KimRass/BERT)|Devlin et al.|[✓] Model architecture<br>[✓] Masked language modeling<br>[✓] BookCorpus data pre-processing<br>[✓] SQuAD data pre-processing<br>[✓] SWAG data pre-processing|
||[Sentence-BERT](https://github.com/KimRass/Sentence-BERT)|Reimers et al.|[✓] Classification loss<br>[✓] Regression loss<br>[✓] Constrastive loss<br>[✓] STSb data pre-processing<br>[✓] WikiSection data pre-processing<br>[✗] NLI data pre-processing|
||[RoBERTa](https://github.com/KimRass/RoBERTa)|Liu et al.|[✓] BookCorpus data pre-processing<br>[✓] Masked language modeling<br>[✗] BookCorpus data pre-processing<br>(SEGMENT-PAIR + NSP)<br>[✗] BookCorpus data pre-processing<br>(SENTENCE-PAIR + NSP)<br>[✓] BookCorpus data pre-processing<br>(FULL-SENTENCES)<br>[✗] BookCorpus data pre-processing<br>(DOC-SENTENCES)|
|2021|[Swin Transformer](https://github.com/KimRass/Swin-Transformer)|Liu et al.|[✓] Patch partition<br>[✓] Patch merging<br>[✓] Relative position bias<br>[✓] Feature map padding<br>[✓] Self-attention in non-overlapped windows<br>[✗] Shifted Window based Self-Attention|
|2024|[RoPE](https://github.com/KimRass/RoPE)|Su et al.|[✓] Rotary Positional Embedding|
|Vision-Language|
|2021|[CLIP](https://github.com/KimRass/CLIP)|Radford et al.|[✓] Training on Flickr8k + Flickr30k<br>[✓] Zero-shot classification on ImageNet1k (mini)<br>[✓] Linear classification on ImageNet1k (mini)|
<!-- ||[PixelLink](https://github.com/KimRass/PixelLink)|Deng et al.|[✓] Model architecture<br>[✓] Instance-balanced cross entropy loss<br>[✓] Model output post-processing| -->

<!-- # 1. Resume
[RESUME](https://github.com/KimRass/KimRass/blob/main/RESUME.md) -->

<!-- # 2. Github Stats
![KimRass' GitHub stats](https://github-readme-stats.vercel.app/api?username=KimRass&show_icons=true&theme=radical) -->
<!-- 
# 2. 'Baekjoon Online Judge' Solved Rank
![hyp3rflow's solved.ac stats](https://github-readme-solvedac.hyp3rflow.vercel.app/api/?handle=rmx1000)
 -->
