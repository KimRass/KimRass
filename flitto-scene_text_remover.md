# 1. 문제 정의
- 이미지에서 텍스트를 지우는 모델입니다.

- 이미지에서 텍스트를 탐지하고 (scene text detection) 자연스럽게 제거하는 (image inpainting) 모델.

# 2. 문제 해결

## 1) 텍스트 중 일부만을 제거
- 이미지에서 텍스트를 지우는 것은 scene text removal 태스크에 해당합니다.
- 그러나 이미지 내의 모든 텍스트를 지우는 보통의 경우와는 다르게, 메뉴 이미지에서는 가게 이름이나 음식의 가격 등 일부의 텍스트만을 지워야 합니다.
- 이러한 요구사항에 맞는 모델을 만들기 위해 문자 단위의 scene text detection 모델을 사용하고자 했습니다.
- 모델의 가중치가 공개되어 있으며 널리 알려진 'CRAFT' 모델 [2]을 사용했습니다.
- 사용자가 제거하고 하는 텍스트를 문자 단위로 지정하는 기능 구현.

## 2) 프로세스
- [1]에서 제시한 방법론을 참고해 크게 3단계 (Scene text detection → Text stroke mask generation → Image inpainting)에 걸쳐 작동하도록 개발했습니다.

### (1) Scene Text Detection
- 'CRAFT'의 출력을 시각화한 이미지입니다. 빨간색에 가까울수록 그 픽셀이 문자의 중심에 가까움을 의미합니다. 이를 'text region score map'이라고 부르겠습니다.
   |Text Region Score Map|
   |-|
   |<img src="https://i.imgur.com/N2zFzGN.png" width="600">|

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
- 'text stroke mask'가 텍스트를 완전히 덮지 못하면 텍스트가 깔끔하게 지워지지 않으므로 dilation을 통해 이를 보정합니다.
- 'text stroke mask'에 watershed를 적용해 문자를 서로 구분하는 'text region segmentation map'을 생성합니다.
   |Text Region Segmentation Map|
   |-|
   |<img src="https://i.imgur.com/3m2TOkK.png" width="600">|
- 'text region score map'을 사용해 각 문자의 중심을 추출합니다.
    |Pseudo Character Centers (PCCs)|
    |-|
    |<img src="https://i.imgur.com/8ZC9zD4.png" width="600">|
- 'text stroke mask'를 둘로 분할하여, 지울 텍스트를 위한 마스크 ('removal mask')와 지우지 않을 텍스트를 위한 마스크 ('revival mask')를 각각 생성합니다.
    |지울 텍스트 ('removal mask')|지우지 않을 텍스트 ('revival mask')|
    |-|-|
    |<img src="https://i.imgur.com/3L5lQT1.png" width="400">|<img src="https://i.imgur.com/mvNOGF3.png" width="400">|

### (3) Image Inpainting
- 사전 학습된 'LaMa' [5]를 사용해 image inpainting을 수행합니다. 다른 어떤 모델을 사용해도 됩니다.
- 'removal mask'와 'revival mask'를 모두 사용해 텍스트를 지운 후 'revival mask'를 사용해 지우지 않을 텍스트를 되살립니다.
- 그 이유는 'removal mask'만을 사용할 경우, 모델이 지워지지 않은 텍스트를 문맥으로 하여 텍스트와 비슷한 텍스쳐를 갖는 이미지를 생성하는 경우가 종종 있었기 때문입니다.
    |텍스트가 지워진 이미지|
    |-|
    |<img src="https://i.imgur.com/V9FbGtR.png" width="600">|

<!-- ## Text Detectability
- Requirements:
    - 해상도와 무관하게 평가가 이루어져야 합니다.
    - 하나의 이미지에서 많은 텍스트를 지울수록 높게 평가되어야 합니다.
- Emplementation
    - Region score map from original image|Region score map from text-removed image
        - <img src="https://i.imgur.com/MN9nz7i.jpg" width="800">
    - 텍스트를 완전하게 제거했음에도 불구하고 기존에 텍스트가 존재하지 않았던 영역에 대해서 쌩뚱맞게 Text detection을 해 버리고 말았습니다.
    - Masked region score map from original image|Masked region score map from text-removed image
        - <img src="https://i.imgur.com/9FKM5Nq.jpg" width="800">
    - 기존에 텍스트가 존재했던 영역에 대한 마스크를 생성하여 Scene text removal 전후 각각의 Region score map을 마스킹합니다. 이로써 새롭게 불필요하게 탐지된 텍스트를 평가 대상에서 제외할 수 있습니다.
    - Scene text removal 전후 각각의 Region score map에 대해서 모든 픽셀에 대한 Region score의 합의 비율을 구하고 이를 1에서 빼 값으로 평가합니다. -1에서 1 사이의 값을 가지며 높을수록 텍스트를 완전하게 제거한 것입니다. -->

# 3. 성과
- RabbitMQ 기반 비동기 AMQP 서버 개발 및 백 오피스와 연동.
- Slack과 연동하여 시스템 에러에 대한 실시간 알림 구현.
- 기존 포토샵을 통한 수작업 대비 처리 속도 개선.
- 'O'사 서비스 대비 뛰어난 시각적 품질 구현.
- 밈 번역 커뮤니티 ['Desert Fox'](https://www.desertfox.io/ko)에 도입.

## 1) Examples

### (1) Success Cases
- 빛의 반사가 많은 경우
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/chinese_character_detection/430_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/chinese_character_detection/430_1_texts_removed.jpg" width="400">|
- 텍스트간 거리가 좁음
  |원본|텍스트 제거|
  |-|-|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/dense_text/539_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/dense_text/539_1_text_removed.jpg" width="400">|
- 복잡한 배경 이미지
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
   <!--|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/374_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/success_cases/texts_in_constrained_space/374_1_texts_removed.jpg" width="400">|-->
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/complicated_background/984_3_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/complicated_background/984_3_texts_removed.jpg" width="400">|
  |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/texts_in_constrained_space/966_1_original.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Scene-Text-Remover/examples/failure_cases/texts_in_constrained_space/966_1_texts_removed.jpg" width="400">|
- 복잡한 배경 이미지
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

## 2) Why Scene Text Detection
- scene text detection 과정이 꼭 필요한가? 즉 bounding box를 그대로 mask로 사용해 image inpainting을 하면 어떻게 되는지 확인해보겠습니다.

|Case 1|Case 2|
|-|-|
|<img src="https://github.com/KimRass/Flitto-ML/assets/67457712/f9cbb17c-5787-49e0-bc00-dd02773aa765" width="500">|<img src="https://github.com/KimRass/Flitto-ML/assets/67457712/ac09af87-a635-4c1b-8b6d-591f1ccc171c" width="400">|
|'한우 육회비빔밥 소반': 타겟 텍스트를 잘 지웠으나 boundig box가 외국어 나란히쓰기 텍스트까지 지워버렸습니다.<br>(외국어 나란히쓰기 텍스트를 지우지 않고자 하는 경우)|이미지에서 너무 넓은 영역을 지우고 생성하려고 하면서 질감을 자연스럽게 살리지 못했습니다.|
|'일일 20개 한정': bounding box가 원본 이미지에 있던 빨간색 직사각형과 겹침이 발생하여 이를 깔끔하게 지우지 못했습니다.||

<!-- # 4. Evaluation
- (기존의 evaluation metric은 이미지 내 텍스트 중 일부만을 지우는 use case에는 적합하지 않을 수 있습니다.)
- On scene text removal task [1] using [7] test set (813 images)
  |Approach|PSNR (↑)|SSIM (↑)|MSE (↓)|
  |-|-|-|-|
  |rule-based approach|34.22|0.9686|0.0017|
  |learning-based approach|32.95|0.9536|0.0019|-->

# 4. References
- [1] [Erasing Scene Text with Weak Supervision](https://github.com/KimRass/KimRass/blob/main/Flitto/papers/erasing_scene_text_with_weak_supervision.pdf)
- [2] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [3] [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRF](https://github.com/KimRass/KimRass/blob/main/Flitto/papers/semantic_image_segmentation_with_deep_convolutional_nets_and_fully_connected_crfs.pdf)
- [4] [Stroke-Based Scene Text Erasing Using Synthetic Data for Training](https://github.com/KimRass/KimRass/blob/main/Flitto/papers/stroke_based_scene_text_erasing_using_synthetic_data_for_training.pdf)
- [5] [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
- [6] [KAIST Scene Text Database](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)
- [7] [SCUT-EnsText](https://github.com/HCIILAB/SCUT-EnsText)
<!-- - [8] [CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://github.com/KimRass/KimRass/tree/main/Flitto/papers/cleval_character_level_evaluation_for_text_detection_and_recognition_tasks.pdf) -->
