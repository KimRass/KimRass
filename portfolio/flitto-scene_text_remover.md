# 1. 문제 정의
- 이미지 번역에서는 텍스트를 제거하고 번역된 내용을 렌더링하는 과정이 필요합니다.
- 텍스트를 자연스럽게 제거하기 위해, 기존에는 디자인팀이 포토샵을 사용하여 정교한 수작업을 진행했습니다. 이 과정은 많은 시간과 노력을 요구하며, 작업자에게 피로감을 초래했습니다. 또한, 서비스를 상용화하기에는 수지타산이 전혀 맞지 않았습니다.
- 이에 image inpainting 기술을 활용한 자동화된 모델을 도입하고자 개발에 착수했습니다.

# 2. 문제 해결

## 1) 텍스트 중 일부만을 제거
- 이미지에서 텍스트를 제거하는 것은 scene text removal 태스크에 해당합니다.
- 그러나 이미지 내의 모든 텍스트를 제거하는 보통의 경우와는 다르게, 메뉴 이미지에서는 가게 이름이나 음식의 가격 등 일부의 텍스트만을 제거해야 합니다.
- 이러한 요구사항에 맞는 모델을 만들기 위해 문자 단위의 scene text detection 모델을 사용하고자 했습니다.
- 모델의 가중치가 공개되어 있으며 널리 알려진 'CRAFT' [2]을 사용했습니다.
- 사용자가 제거하고 하는 텍스트를 문자 단위로 지정하는 기능 구현.

## 2) 프로세스
- 이미지에서 텍스트를 탐지하고 (scene text detection) 이를 바탕으로 제거할 영역을 정교하게 계산하여 (text stroke mask generation) 텍스트만 자연스럽게 제거하는 (image inpainting) 3단계로 작동합니다 [1].

### (1) Scene Text Detection
- 'CRAFT'의 출력을 시각화한 이미지입니다. 빨간색에 가까울수록 그 픽셀이 문자의 중심에 가까움을 의미합니다. 이를 'text region score map'이라고 부르겠습니다.
   |Text Region Score Map|
   |-|
   |<img src="https://i.imgur.com/N2zFzGN.png" width="600">|

- scene text detection이 반드시 필요한가:
  |scene text detection|바운딩 박스|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/resources/GfnExpj_d.png" width="200">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/resources/2nMPnO1_d.webp" width="200">|
  ||'일일 20개 한정'의 바운딩 박스가 빨간색 직사각형보다 크게 생성되어 텍스트가 깔끔하게 제거하지 못했습니다.|
  ||'한우 육회비빔밥 소반' 텍스트는 잘 제거했으나 바운딩 박스가 그 밑의 영어 텍스트까지 제거해 버렸습니다.|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/resources/4DeWz3L_d.png" width="200">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/resources/2XjZyqR_d.png" width="200">|
  ||이미지에서 너무 많은 픽셀을 재생성하려고 했기 때문에 질감이 뭉게졌습니다.|

### (2) Text Stroke Mask Generation
- 규칙 기반 및 학습 기반 방법론을 모두 개발했습니다.
- 규칙 기반 방법론:
    - adaptive image thresholding 및 connected-component labeling (CCL)을 통해 이미지를 아주 작게 조각냅니다.
    |조각난 이미지|
    |-|
    |<img src="https://i.imgur.com/ujrxsrk.png" width="600">|
    |이해를 돕기 위해 26개의 색상으로 단순화|
    - 이 조각들과 'text region score map'과의 겹침을 계산하여 'text stroke mask'를 생성합니다.
    - fully-connected CRF [3]을 통해 'text stroke mask'를 좀 더 세밀하게 보정합니다.
- 학습 기반 방법론:
    - scene text segmentation 태스크에 대해 사전 학습된 모델 [4]을 사용합니다.
    - 각 바운딩 박스를 종횡비가 1:5 또는 5:1이 되도록 분할하거나 패딩을 추가하고, 640 × 128의 해상도가 되도록 크기를 조정하여 모델에 입력합니다.
    - 모델의 출력을 원본 이미지의 원래 위치에 삽입하여 'text stroke mask'를 생성합니다.
    - fully-connected CRF [3]을 통해 'text stroke mask'를 좀 더 세밀하게 보정합니다.
- 'text stroke mask'가 텍스트를 완전히 덮지 못하면 텍스트가 깔끔하게 제거되지 않으므로 dilation을 통해 이를 보정합니다.
- 'text stroke mask'에 watershed를 활용하여 서로 다른 문자를 구분하는 'text region segmentation map'을 생성합니다.
   |Text Region Segmentation Map|
   |-|
   |<img src="https://i.imgur.com/3m2TOkK.png" width="600">|
- 'text region score map'을 사용해 각 문자의 중심 (Pseudo Character Center (PCC) [8])을 추출합니다.
    |Pseudo Character Centers (PCCs)|
    |-|
    |<img src="https://i.imgur.com/8ZC9zD4.png" width="600">|
- 'text stroke mask'를 둘로 분할하여, 제거할 텍스트를 위한 마스크 ('마스크1')와 제거하지 않을 텍스트를 위한 마스크 ('마스크2')를 각각 생성합니다.
    |'마스크1'|'마스크2'|
    |-|-|
    |<img src="https://i.imgur.com/3L5lQT1.png" width="400">|<img src="https://i.imgur.com/mvNOGF3.png" width="400">|

### (3) Image Inpainting
- 사전 학습된 'LaMa' [5]를 사용해 image inpainting을 수행합니다. 다른 어떤 모델을 사용해도 됩니다.
- '마스크1'와 '마스크2'를 모두 사용해 텍스트를 지운 후 '마스크2'를 사용해 제거하지 않을 텍스트를 되살립니다.
- 그 이유는 '마스크1'만을 사용할 경우, 모델이 제거되지 않은 텍스트를 문맥으로 하여 텍스트와 비슷한 텍스쳐를 갖는 이미지를 생성하는 경우가 종종 있었기 때문입니다.
    |텍스트가 제거된 이미지|
    |-|
    |<img src="https://i.imgur.com/V9FbGtR.png" width="600">|

### (4) 모델 평가
- 규칙 기반 방법론과 학습 기반 방법론 각각에 대하여, [7]을 사용해 이미지의 모든 텍스트를 제거하고 정답 이미지와 비교했습니다. (이 방법은 일부의 텍스트만을 제거하는 실제 사용 사례를 정확히 반영하지는 않습니다.)
  |방법론|PSNR (↑)|SSIM (↑)|MSE (↓)|
  |-|-|-|-|
  |규칙 기반|34.22|0.9686|0.0017|
  |학습 기반|32.95|0.9536|0.0019|
- 이처럼 규칙 기반 방법론의 성능이 약간 더 높았습니다. 또한 사용자도 규칙 기반 방법론이 더 뛰어난 것 같다는 의견을 주어, 규칙 기반 방법론을 도입하기로 결정했습니다.

# 3. 성과
- 기존의 포토샵을 통한 수작업 대비 처리 속도가 대폭 향상됐습니다.
- 시각적 품질에 있어서, 경쟁사인 'O'사의 서비스 대비 뛰어난 결과를 보여줍니다.

## 1) 예시

### (1) Success Cases
- 빛 반사
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/chinese_character_detection/430_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/chinese_character_detection/430_1_texts_removed.jpg" width="400">|
- 밀도 높은 텍스트
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/dense_text/539_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/dense_text/539_1_text_removed.jpg" width="400">|
- 복잡한 배경
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/complicated_background/1555_5943_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/complicated_background/1555_5943_text_removed.jpg" width="400">|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/complicated_background/696_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/complicated_background/696_1_texts_removed.jpg" width="400">|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/complicated_background/389_2_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/complicated_background/389_2_text_removed.jpg" width="400">|
- 고해상도 이미지
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/high_resolution/ontheboarder1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/high_resolution/ontheboarder1_texts_removed.png" width="400">|
- 한정된 공간 내의 텍스트
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/gajok_original.png" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/gajok_text_removed.png" width="400">|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/776_2846_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/776_2846_text_removed.jpg" width="400">|

### (2) Failure Cases
- 텍스트 탐지 실패
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/chinese_character_detection/400_2_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/chinese_character_detection/400_2_texts_removed.jpg" width="400">|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/partial_removal/1010_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/partial_removal/1010_1_texts_removed.jpg" width="400">|
- 한정된 공간 내의 텍스트
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/complicated_background/984_3_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/complicated_background/984_3_texts_removed.jpg" width="400">|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/texts_in_constrained_space/966_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/texts_in_constrained_space/966_1_texts_removed.jpg" width="400">|
- 복잡한 배경
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/complicated_background/958_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/complicated_background/958_1_texts_removed.jpg" width="400">|
- 저해상도 텍스트
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/low_text_readability/927_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/low_text_readability/927_1_texts_removed.jpg" width="400">|
- 불완전한 텍스트 제거
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/partial_removal/985_4_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/partial_removal/985_4_texts_removed.jpg" width="400">|
   <!--|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/374_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/374_1_texts_removed.jpg" width="400">|-->

# 4. References
- [1] [Erasing Scene Text with Weak Supervision](https://ieeexplore.ieee.org/document/9093544)
- [2] [Character Region Awareness for Text Detection](https://arxiv.org/abs/1904.01941)
- [3] [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRF](https://arxiv.org/abs/1412.7062)
- [4] [Stroke-Based Scene Text Erasing Using Synthetic Data for Training](https://ieeexplore.ieee.org/document/9609970)
- [5] [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
<!-- - [6] [KAIST Scene Text Database](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database) -->
- [7] [SCUT-EnsText](https://github.com/HCIILAB/SCUT-EnsText)
- [8] [CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://arxiv.org/abs/2006.06244)
