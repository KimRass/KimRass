- 프로젝트 소개:
    - 초고해상도·초고품질 영상에 대한 video super-resolution 모델 개발.
- 프로젝트 진행:
    - PM 1명, 리서치 엔지니어 1명 (본인) 참여.
- 업무 내용:
    - 연구 및 실험:
        - latent vector 시각화하여 이미지의 해상도에 따른 원본 이미지와 latent vector 사이의 상관관계 확인.
        - latent vector normalization 유무에 따른 모델 학습 속도와 성능 비교.
        - latent vector의 국소 영역 및 특정 채널에 대한 조작이 재구축 이미지에 미치는 영향 분석.
        - 최신 Autoencoder 모델의 이미지 재구축 성능을 정량적으로 비교평가하여 최적의 모델 선정.
        - 'BasicVSR++' 모델을 기반으로 Autoencoder를 결합하여 latent space 도입, 계산 효율성 향상.
        - optical flow 후처리 방법이 video super-resolution 성능에 미치는 영향 비교.
    - 모델 학습 최적화:
        - 학습 데이터를 사전 전처리하여 파일 I/O 최소화 및 GPU 활용도와 학습 속도 최적화.
        - 8대의 RTX 4090과 `accelerate` 라이브러리를 활용하여 데이터 분산 학습 수행.
    - TIPS (Tech Incubator Program for Startup) 프로그램의 기술 부문 발표 자료 작성.



# 1. 문제 정의

# 2. 문제 해결

# 3. 성과

# 4. References



# 1. Autoencoders Image Reconstruction Evaluation ('eval')

## 1) Objective
- Autoencoder 모델들을 서로 정량적으로 비교하여 본 연구에 적합한 모델을 선별합니다.

## 2) Methodology
- 3개의 데이터셋 (REDS, Vid4, Vimeo-90K)에 대해서, 이미지를 Encode → Decode하여 재구축합니다 (Reconstructed image). 이를 원본 이미지 (Input image)와 비교하여 차이를 계산합니다.
- Input image의 Spatial size가 8의 배수가 아닐 경우 8의 배수가 되도록 자동으로 이미지를 Crop합니다 (e.g., Encoding 후 Decoding 시 (129, 258) → (128, 256)). 따라서 Input image보다 Reconstructed image가 작을 수 있으며 평가 시에는 Spatial size을 동일하게 맞추기 위해 Input image의 일부를 Reconstructed image에 맞게 Crop합니다.
    - *8의 배수가 되도록 자동으로 Crop되는 과정:
        1. `(0, 1, 0, 1)`  만큼 Pad.
        2. `nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2))`  통과 (`in_channels`, `out_channels`만 맞으면 에러는 발생하지 않음).
    - 예를 들어 Encoding시에 Input image의 Spatial size가 (65, 71)이라고 하면 '(65, 71) → (66, 72) → (32, 35) → (33, 36) → (16, 17) → (17, 18) → (8, 8)'의 과정을 거쳐서 1/8로 Downsample됩니다 (Encoding). (Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/downsampling.py#L141-L147)
- Input image와 Reconstructed image 각각을 `$[0, 255]$`의 범위를 갖고 `np.float64`의 `dtype`을 갖는 NumPy array로 변환했습니다.

### (1) Models
- SD-VAE-FT-MSE:
    - `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`
    - https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt
- SDXL-VAE:
    - `AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")`
    - 'SD-VAE-FT-MSE'와 Architecture는 완전히 동일하고, 차이점은 학습 시 더 큰 Batch size (256 vs. 9)와 EMA를 사용했다는 점입니다.
- SD3-Medium:
    - `DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers").vae`
    - 'SD-VAE-FT-MSE'와의 차이점은 Latent vector의 Channel dimension이 4에서 16으로 증가했다는 점과 Encoder와 Decoder 사이에 2개의 Point-wise convolutional layers가 사용되지 않는다는 점입니다. 파라미터 수는 83,819,683 vs. 83,653,863으로 크게 다르지 않습니다.

### (2) Datasets
| Data-set | 데이터 개수 | 해상도 |
| --- | --- | --- |
| REDS validation set | 3,000 | 1280 × 720 |
| Vid4 test set | 171 | 720 x 480 ('walk', 'foliage') |
|  |  | 704 x 576 ('city') |
|  |  | 720 x 576 ('calender') |
| Vimeo-90K test set | 54,768 | 448 × 256 |
| All | 57,939 | - |

### (3) Metrics
- 'MSE (↓)': The mean of the pixel-wise squared errors. (The standard deviation of the pixel-wise squared errors.)
- 'STD of squared error (↓)': The standard deviation of the pixel-wise squared errors.
- 'RMSE (↓)': The root of the mean of the pixel-wise squared errors. (The root of the standard deviation of the pixel-wise squared errors.)
- 'Root STD of squared error (↓)': The root of the standard deviation of the pixel-wise squared errors.
- 'LPIPS (↓)': The mean of the Learned Perceptual Image Patch Similarity (LPIPS). (The standard deviation of the Learned Perceptual Image Patch Similarity (LPIPS).)

## 3) Results
- 모델: 'SD3-Medium' (`DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers").vae`)
    - 'SD-VAE-FT-MSE'와의 차이점은 Latent vector의 Channel dimension이 4에서 16으로 증가했다는 점과 Encoder와 Decoder 사이에 2개의 Point-wise convolutional layers가 사용되지 않는다는 점입니다. 파라미터 수는 83,819,683 vs. 83,653,863으로 크게 다르지 않습니다.

    | Dataset | MSE (↓) | STD of squared error (↓) | RMSE (↓) | Root STD of squared error (↓) | LPIPS (↓) |
    | --- | --- | --- | --- | --- | --- |
    | REDS | 65.88 (± 38.27) | 170.64 | 7.82 (± 2.19) | 12.55 | 0.0329 (± 0.0143) |
    | Vid4 | 146.61 (± 109.01) | 360.00 | 11.27 (± 4.44) | 17.60 | 0.0426 (± 0.0190) |
    | Vimeo-90K | 26.92 (± 32.22) | 97.42 | 4.64 (± 2.31) | 8.67 | 0.0119 (± 0.0092) |
    | REDS + Vid4 + Vimeo-90K | 79.80 (± 59.83) | 209.35 | 7.91 (± 2.98) | 12.94 | 0.0291 (± 0.0142) |

# 2. Latent Vector Visualization ('vis')

## 1) Methodology
- Colormap은 'Jet'입니다.
- Latent vector의 Spatial size는 Input image의 Spatial size 대비 1/8이므로 Latent vector를 8배로 Nearest upsample했습니다.
- Input image의 Spatial size가 8의 배수가 아닐 경우 8의 배수가 되도록 자동으로 이미지를 Crop합니다 (e.g., (129, 258) → (128, 256)). 따라서 Input image를 제외한 나머지 7개의 이미지는 오른쪽 또는 아래에 검은색의 Padding이 존재할 수 있습니다. Padding이 검은색인 이유는 Canvas를 검은색으로 했기 때문입니다.
- REDS, UDM10, Vid4, Vimeo-90K 의 데이터 중에서 BasicVSR, BasicVSR++ 논문에서 정성적 비교를 위해 사용한 일부만을 대상으로 했습니다.

# 3. Research on Stable Diffusion 3 VAE ('architecture')

## 1) Paper Analysis
- Autoencoder의 성능이 Text-to-image 모델의 상한선을 결정합니다. [1-5.2.1]
- (다른 파라미터들은 모두 동일하게 둔 채) Latent channels의 수를 4→8→16으로 점차 증가시킴에 따라 Autoencoder의 이미지 재구축 성능이 상승합니다 (FID, Perceptual similarity, SSIM, PSNR). [1-5.2.1, 2-3.1]
- Latent channels의 수를 32로 더 증가시키면 더 디테일을 잘 포착합니다. [2-3.1] (이 Autoencoder를 사용하지 않은 이유는 아마도 Diffusion model을 너무 큰 것을 사용해야만 하기 때문인 것 같습니다.)
- Latent channels의 수가 클수록 (16 vs. 8) 동일한 Text-to-image generation 성능을 내기 위해 더 큰 Diffusion model이 필요합니다. Diffusion model의 크기가 동일하다면 16-channels autoencoder를 사용하는 것이 8-channels autoencoder을 사용하는 것보다 Text-to-image generation 성능이 낮습니다. [1-B.2]
- 4-channel autoencoder는 그 압축률이 너무 높기 때문에 디테일을 복원하지 못하는데 특히 작은 오브젝트에 대해서 이 점이 두드러집니다. [2-3.1]
- Adversarial loss를 적용하고 규칙 기반의 Fourier feature transform을 원본 이미지에 적용하는 전처리를 적용해 원본 이미지를 더 고차원으로 변환함으로써 디테일을 잘 포착하도록 하면 Autoencoder의 성능이 향상됩니다. [2-3.1]

## 2) Code Analysis
- 'SD-VAE-FT-MSE', 'SDXL-VAE'와의 차이점은 Latent channels의 수가 16 vs. 4라는 점과 Encoder와 Decoder 사이에 추가적인 2개의 Convolutional layer (`quant_conv`, `post_quant_conv`)가 미존재 vs. 존재라는 점뿐입니다.

    | Layer | Shape ('SD3-Medium') | Shape ('SD-VAE-FT-MSE') |
    | --- | --- | --- |
    | encoder.conv_out.weight  | (32, 512, 3, 3) | (8, 512, 3, 3) |
    | encoder.conv_out.bias  | (32,) | (8,) |
    | decoder.conv_in.weight  | (512, 16, 3, 3) | (512, 4, 3, 3) |
    | quant_conv.weight | - | (8, 8, 1, 1) |
    | quant_conv.bias | - | (8,) |
    | post_quant_conv.weight  | - | (4, 4, 1, 1) |
    | post_quant_conv.bias | - | (4,) |
- Architecture after the encoder
    - Encoder 이후에 (Skip connection에 사용된 2개를 제외하고) 총 33개의 Covolutional layers와 1개의 Attention이 사용되었습니다.
- 'architecture-SD3-Medium.png'

| 1 | decoder.conv_in  | nn.Conv2d(16, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| --- | --- | --- | --- |
| 2 | decoder.mid_block.resnets[0].conv1  | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 3 | decoder.mid_block.resnets[0].conv2  | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| - | decoder.mid_block.attentions[0]  | Attention |  |
| 4 | decoder.mid_block.resnets[1].conv1  | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 5 | decoder.mid_block.resnets[1].conv2  | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 6 | decoder.up_blocks[0].resnets[0].conv1 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 7 | decoder.up_blocks[0].resnets[0].conv2 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 8 | decoder.up_blocks[0].resnets[1].conv1 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 9 | decoder.up_blocks[0].resnets[1].conv2 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 10 | decoder.up_blocks[0].resnets[2].conv1 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 11 | decoder.up_blocks[0].resnets[2].conv2 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 12 | decoder.up_blocks[0].upsamplers[0].conv | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 13 | decoder.up_blocks[1].resnets[0].conv1 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 14 | decoder.up_blocks[1].resnets[0].conv2 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 15 | decoder.up_blocks[1].resnets[1].conv1 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 16 | decoder.up_blocks[1].resnets[1].conv2 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 17 | decoder.up_blocks[1].resnets[2].conv1 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 18 | decoder.up_blocks[1].resnets[2].conv2 | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 19 | decoder.up_blocks[1].upsamplers[0].conv | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 20 | decoder.up_blocks[2].resnets[0].conv1 | nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 21 | decoder.up_blocks[2].resnets[0].conv2  | nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 22 | decoder.up_blocks[2].resnets[1].conv1 | nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 23 | decoder.up_blocks[2].resnets[1].conv2  | nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 24 | decoder.up_blocks[2].resnets[2].conv1 | nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 25 | decoder.up_blocks[2].resnets[2].conv2 | nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 26 | decoder.up_blocks[2].upsamplers[0].conv | nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 27 | decoder.up_blocks[3].resnets[0].conv1 | nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 28 | decoder.up_blocks[3].resnets[0].conv2  | nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 29 | decoder.up_blocks[3].resnets[1].conv1 | nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 30 | decoder.up_blocks[3].resnets[1].conv2  | nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 31 | decoder.up_blocks[3].resnets[2].conv1 | nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 32 | decoder.up_blocks[3].resnets[2].conv2 | nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |
| 33 | decoder.conv_out | nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) |  |

## 3) References
- [1] https://arxiv.org/abs/2403.03206
- [2] https://arxiv.org/abs/2309.15807

# 4. Latent Vector Horizontal Flip ('hflip')
- 원본 이미지의 Latent vector와 좌우 반전시킨 이미지의 Latent vector가 서로 좌우 반전의 관계에 있지 않을까 생각했으나 사실은 조금 달랐습니다. 좌우 반전시킨 Latent vector를 디코더에 입력해서 생성한 이미지는 원본 이미지와 전체적인 구조는 유사하나 이미지를 이루는 각각의 선들이 좌우 반전된 듯한 모습이었습니다.
    - '저해상도 원본.png'
    - '저해상도 Latent vector 좌우 반전.png'
    - '고해상도 원본.png'
    - '고해상도 Latent vector 좌우 반전.png'

# 5. LatentBasicVSR++ - Optical Flow Processing

## 1) Methodology 1
- 원래의 BasicVSR++와 동일하게 저해상도 비디오를 SPyNet에 입력하여 Optical flow를 추출한 후 1/8로 Bilinear downsampling을 적용합니다.
- Optical flow에 대해서 Bilinear downsampling을 적용한다는 것이 개념적으로 맞지 않다고 할 수 있습니다.
- Loss가 0.3 부근에서 거의 줄어들지 않는 현상이 존재합니다.

## 2) Methodology 2
- 원래의 BasicVSR++와 동일하게 저해상도 비디오를 가지고 SPyNet을 통해 Optical flow를 추출한 후 `kernel_size=8, stride=8`을 가지고 2d average pooling을 적용합니다.
- Loss가 0.3 부근에서 거의 줄어들지 않는 현상이 존재합니다.

## 3) Methodology 3
- 저해상도 비디오의 Latent vector를 SPyNet에 입력하여 Optical flow를 추출합니다.
- 저해상도로서 64 × 64를 사용할 경우 Latent vector는 8 × 8의 해상도를 갖습니다. SPyNet의 내부 알고리즘에 있어서 최소 64 × 64 이상의 해상도만 처리가 가능하므로 이 방법으로는 Latent vector를 처리할 수 없습니다.

## 4) Methodology 4
```python
self.flow_conv = nn.Sequential(
    nn.Conv2d(2, flow_conv_channels, 3, 2, 1, bias=True),
    nn.LeakyReLU(negative_slope=0.1),
    nn.Conv2d(
        flow_conv_channels, flow_conv_channels, 3, 2, 1, bias=True,
    ),
    nn.LeakyReLU(negative_slope=0.1),
    nn.Conv2d(flow_conv_channels, 2, 3, 2, 1, bias=True),
)
```
- Loss가 0.3 부근에서 거의 줄어들지 않는 현상이 존재합니다.

## 5) Methodology 5
- SPyNet을 통해 Optical flow를 추출하지 않습니다.
- 원래의 BasicVSR++는 저해상도 비디오를 SPyNet에 입력해서 Optical flow를 추출하고, 저해상도 비디오로부터 추출된 특징을 가지고 이 Optical flow와 연산을 진행하도록 되어 있었습니다. LatentBasicVSR++에서는 저해상도 비디오를 SPyNet에 입력해서 Optical flow를 추출하고, 저해상도 비디오의 Latent vector로부터 추출된 특징을 가지고 이 Optical flow와 연산을 진행하도록 했습니다. 그런데 Optical flow는 저해상도 비디오 자체에서 추출되었고 특징은 저해상도 비디오의 Latent vector로부터 추출되었기 때문에 이 둘이 존재하는 Space가 서로 달라 Align이 되지 않는 것이 Loss가 줄어들지 않는 원인이라고 생각했습니다.

# 6. Data Preprocessing

## 1) On-the-fly Image Cropping
- `__getitem__` 에서 64 × 64로 무작위로 이미지를 자르고 확률적으로 좌우 반전을 적용합니다.
- 하나의 원본 이미지에서 거의 모든 부분을 다 모델 학습에 사용할 수 있습니다.
- GPU 사용률이 크게 요동칩니다. 데이터를 불러오는 과정에서 GPU가 놀고 있는 것으로 추측됩니다.

## 2) Pre-cropping Images
- Sliding window로 64 × 64로 이미지를 미리 잘라 저장해두고 학습 시 사용합니다. `__getitem__` 에서는 확률적으로 좌우 반전만 적용합니다.
- `--lq_img_size=64`, `--stride=64`를 사용할 경우 REDS에 대해서 1 epoch의 데이터 크기가 10배가 됩니다.
- 원본 이미지에서 정해진 영역만이 모델 학습에 사용됩니다.
- GPU 사용률이 거의 항상 100%를 기록합니다.

# 7. Training on REDS ('training_on_reds')

## 1) Data Pre-processing
- 연속된 프레임의 수: 3
- LQ 이미지 사이즈: 64, HQ 이미지 사이즈: 256
- LQ 이미지는 원본 해상도가 180 × 320인데 VAE를 사용한 ×1/8 압축과 4× VSR을 고려하여 176 × 320이 되도록 각 프레임의 좌상단을 Crop했습니다.
- 마찬가지로 HQ 이미지는 원본 해상도가 720 × 1280인데 704 × 1280이 되도록 각 프레임의 좌상단을 Crop했습니다.
- Training set
    - LQ: 'train_sharp_bicubic', HQ: 'train_sharp'
    - 240개의 동영상이 존재하고 각각은 100개의 프레임으로 구성됩니다. 각 동영상으로부터 0~2, 1~3, ..., 97~99번째 프레임을 가지고 입력 데이터를 구성했습니다.
    - 64 × 64의 해상도가 되도록 균일분포로 무작위로 Crop했습니다.
    - 50%의 확률로 좌우 반전을 적용했습니다.
- Validation set
    - LQ: 'val_sharp_bicubic', HQ: 'val_sharp'
    - 64 × 64의 해상도가 되도록 Center crop했습니다.

### (1) Off-the-fly Crop
- As-is:
    - `__getitem__()` 내부에서 매번 균일분포의 무작위로 Crop했습니다.
    - 학습 시 File I/O로 인해 Bottleneck으로 작용했습니다.
- To-be:
    - Sliding window 방식으로 이미지에서 $3 \times 5 = 15$개의 패치로

## 2) Training
- VAE는 SD3-Medium의 것을 사용했습니다.
- AdamW optimizer를 사용했습니다 ([`lr=](http://lr=args.lr/)0.001, betas=(0.9, 0.99)`). Scheduler는 `"linear"` 를 사용했다가 `"constant"`로 변경했습니다.)
- `max_norm=3`을 가지고 Gradient clipping을 사용했습니다.
- 1개의 Node에서 4대의 RTX 4090을 사용했습니다. GPU당 Batch size는 16, Gradient accumulation step의 수는 16로 했습니다.
- 1 epoch마다 학습에 약 160초가 소요됐습니다.
- Validation
    - 한 Epoch가 끝난 후 Validation set에 대해 Chabonnier loss를 계산하고 Chabonnier loss를 기준으로 성능이 향상될 때만 Checkpoint를 저장하도록 했습니다. 최대 3개의 최근 Checkpoints를 남기고 오래된 순서대로 삭제하도록 했습니다.

## 3) Results
- 'G.T. image from validation set.png'
- 'Predicted image from validation set (18,000번 학습 후).png'
- 'Predicted image from validation set (24,000번 학습 후).png'
- 'Predicted image from validation set (36,000번 학습 후).png'

# 8. Latent Vector Normalization

## 1) Methodology
- REDS의 Training data를 대상으로 SD3 Medium VAE를 이용해서 이미지들을 Latent vector로 변환했을 때 16개의 채널별로 최소값과 최대값을 조사했습니다.
    ```python
    latent_min = (-2.5693567, -2.3477385, -2.6224306, -2.6576562, -2.863414, -3.8316438, -3.4375908, -4.537225, -1.9074624, -1.457557, -0.9430707, -3.0715818, -1.3099165, -2.394425, -1.8564366, -2.400173)
    latent_max = (4.3139405, 2.651021, 3.201024, 4.3035817, 3.8277922, 3.1345496, 2.8162382, 2.6298013, 3.829821, 3.8353004, 4.8711133, 3.8269386, 3.4687276, 3.1760585, 3.634507, 3.9133532)
    ```
- Encode 후에는 `latent_min`과 `latent_max`를 사용해서 Min-max scale한 후 $[-1, 1]$의 범위를 갖도록 Normalize했으며 Decode 후에는 역연산을 수행했습니다.

## 2) Results
- 학습 속도가 향상되는 효과는 있었지만 Training loss threshold가 존재하는 문제는 여전히 있었습니다.

# 9. Residual Learning ('residual_learning')
- Forward pass에서 가장 끝부분의 알고리즘을 어떻게 하느냐에 관한 다음의 3가지 방법을 비교했습니다.

## 1) No Residual Learning
- Methodology:
    ```python
    h = self.lqs_upsample(h)
    outs.append(h)
    ```
- Results:
    - 1,200 epochs 학습 후의 Charbonnier loss: 0.0289
    - Predicted images are most faithful;
        - 'no_residual_learning.png'

## 2) Residual Learning Using Latent Vector Upsampling
- Methodology:
    ```python
    h = self.lqs_upsample(h)
    h = h + F.interpolate(
        lqs_latent[:, i, :, :, :],
        scale_factor=4,
        mode="bilinear",
        align_corners=False,
    )
    outs.append(h)
    ```
- Results:
    - 1,200 epochs 학습 후의 Charbonnier loss: 0.0468
    - Predicted images are more blurry;
        - 'residual_learning_using_latent_vector_upsampling.png'

## 3) Residual Learning Using Input Image Upsampling
- Methodology:
    ```python
    h = self.lqs_upsample(h)
    h = h + up_lqs_latent[:, i, :, :, :]
    outs.append(h)
    ```
- Results:
    - 1,200 epochs 학습 후의 Charbonnier loss: 0.0357
    - 'residual_learning_using_input_image_upsampling.png'

# 10. Model Hyperparameters
| `flow_conv_channels` | Charbonnier loss after training for 1,200 epochs |
| --- | --- |
| 2 | 0.0303 |
| 4 | 0.0228 |
| 8 | 0.0482 |
| 16 | 0.0412 |
| 32 | 0.0723 |
| 64 | 0.0227 |

| `deform_groups` | Charbonnier loss after training for 1,200 epochs |
| --- | --- |
| 16 | 0.1131 |
| 32 | 0.0233 |
| 64 | 26.3391 |
| 128 | 0.0265 |
| 256 | 0.0998 |
| 512 | 0.0787 |

# 11. Loss Function ('loss_fn')

# 12. Latent Vector Manipulation

## 1) Experiment 1 ('latent_vector_sampling.py')

### (1) Objective
- 이미지를 VAE의 Encoder를 통과시킨 후 Sample하는데 이때 무작위성이 어느 정도 있는지 정량적, 정성적으로 측정합니다.

### (2) Methodology
1. 이미지를 Encoder를 통과시켜 Gaussian distribution을 생성합니다. 이 분포의 평균과 분산을 출력합니다. 이 값들은 변하지 않습니다.
2. 10회에 걸쳐 이 분포로부터 표본 (Latent vector)을 추출합니다.
3. 표본을 Decoder에 넣어 이미지를 재구축하고 Element-wise mean을 출력합니다.

### (3) Results
```
[ Posterior mean: 0.98298 ][ Posterior STD: 0.00091 ]
[ Element-wise mean of reconstructed image: ]
    [ 0.321701 ]
    [ 0.321702 ]
    [ 0.321703 ]
    [ 0.321701 ]
    [ 0.321698 ]
    [ 0.321706 ]
    [ 0.321699 ]
    [ 0.321699 ]
    [ 0.321705 ]
    [ 0.321701 ]
```
- Posterior의 표준편차가 매우 작기 때문에 Reconstructed image에 있어서 변화량 또한 매우 작습니다.

## 2) Experiment 2

### (1) Objective
- Image VAE의 인코더가 출력하는 Latent variables인 mean, std 값의 역할을 알아내는 것

### (2) Methodology
1. 입력 이미지를 인코딩하고, 인코딩 값의 평균과 표준편차를 각각 변화시키면서 디코딩한 출력 이미지를 계산한다.
2. 입력 이미지와 출력 이미지를 정량 비교하여 변화시킨 인코딩값이 미친 영향을 알아낸다.

### (3) Results
- 결과 1: Encoded image의 표준편차(Latent variable 'STD') 변경

    | 인덱스 | 평균 곱 배율 | encoded image (평균 / 표준편차의 평균) | sampled image (평균 / 표준편차의 평균) |
    | --- | --- | --- | --- |
    | 1 | 100 | 0.983 / 0.091 | 0.981 / 3.978 |
    | 2 | 200 | 0.983 / 0.181 | 0.987 / 3.988 |
    | 3 | 300 | 0.983 / 0.272 | 0.983 / 3.990 |
    | 4 | 400 | 0.983 / 0.363 | 0.987 / 4.010 |
    | 5 | 500 |  0.983 / 0.453 | 0.990 / 4.032 |
    | 6 | 600 | 0.983 / 0.544 | 1.000 / 4.042 |
    | 7 | 700 | 0.983 / 0.634 | 0.975 / 4.091 |
    | 8 | 800 | 0.983 / 0.725 | 1.002 / 4.119 |
    | 9 | 900 | 0.983 / 0.816 | 0.995 / 4.137 |
    | 10 | 1000 | 0.983 / 0.906 | 0.958 / 4.196 |
    - 'Latent variable 표준편차(STD)를 수정했을 때 Output image (좌상부터 우하까지 차례대로 1, 4, 7, 10번 출력 이미지).png'
- 결과 2: Encoded image의 평균값(Latent variable 'Mean') 변경
    | 인덱스 | 평균 곱 배율 | encoded image (평균 / 표준편차의 평균) | sampled image (평균 / 표준편차의 평균) |
    | --- | --- | --- | --- |
    | 1 | 0.50 | 0.491 / 0.001 | 0.491 / 1.988 |
    | 2 | 0.61 |  0.601 / 0.001 | 0.601 / 2.430 |
    | 3 | 0.72 | 0.710 / 0.001 | 0.710 / 2.871 |
    | 4 | 0.83 | 0.819 / 0.001 | 0.819 / 3.313 |
    | 5 | 0.94 | 0.928 / 0.001 | 0.928 / 3.755 |
    | 6 | 1.06 | 1.038 / 0.001 | 1.038 / 4.197 |
    | 7 | 1.17 | 1.147 / 0.001 | 1.147 / 4.638 |
    | 8 | 1.27 | 1.256 / 0.001 | 1.256 / 5.080 |
    | 9 | 1.39 | 1.365 / 0.001 | 1.365 / 5.522 |
    | 10 | 1.50 | 1.474 / 0.001 | 1.474 / 5.964 |
    - 'Latent variable 평균(Mean)을 수정했을때 Output Image (좌상 부터 우하까지 차례대로  1번, 4번, 7번, 10번 출력 이미지).png'

## 3) Experiment 3 ('latent_vector_manipulation.py')
- 목적
    - Latent vector에서 일부 영역의 값만 변화시킬 때 재구축된 이미지에서도 그에 해당하는 영역에서는 어떤 변화가 발생하는지 그리고 전체 영역에서는 어떤 변화가 발생하는지 정성적으로 관찰합니다.
- 결과
    - 일부 영역에 대한 조작
        - 국소적인 영역 (`--ltrb="(13, 7, 18, 9)"` )에 대해서 모든 픽셀의 값을 Latent vector의 채널별 평균으로 대체하거나 정규분포에서 샘플링한 값으로 대체할 경우 그 외의 영역에 대해서는 생성된 이미지에 변화가 (거의) 없었습니다.
            - 'std_multi=0.png'
            - 'std_multi=1.png'
        - 그러나 정규분포의 표준편차를 큰 값으로 키워감에 따라 Latent vector에서 조작을 가한 영역 외의 영역에서도 생성된 이미지에서 변화가 점점 커졌습니다.
            - 'std_multi=20.png'
            - 'std_multi=40.png'
            - 'std_multi=80.png'
    - 일부 채널에 대한 조작
        - 국소적인 영역 (`--ltrb="(13, 7, 18, 9)"` )에 대해서 Latent vector의 16개의 채널 중 하나씩만 그 값을 각 채널의 평균으로 대체했습니다.
        - 인덱스가 큰 채널일수록 생성된 이미지에서 더 넓고 분명한 변화가 보입니다.
            - 'channel=0.png'
            - 'channel=5.png'
            - 'channel=8.png'
            - 'channel=15.png'

# 13. BasicVSR++ ('basicvsrpp')
