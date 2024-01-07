- 아랍어를 주로 포함하고 있는 scene text image에 대해서 이미지의 회전된 각도를 0°, 90°, 180°, 270° 중 하나로 분류하는 모델입니다.

# 1. Data
- 자체 크라우드 소싱 데이터 수집 플랫폼을 통해 수집한 이미지 (31,652장) +  EvArEST [1] detection dataset (510장)
- 전체 데이터의 각각 80%, 10%, 10%을 training set, valiation set, test set으로 나눴습니다.
## 1) Data Augmentation
- [3]에서 사용된 data augmentation 기법들을 참고하여 다소 강하게 구현했습니다.
    - (작은 각도로) 이미지 회전
    - 이미지 자르기
    - color jittering (hue, saturation, brightness, contrast) 적용
    - Gaussian blur 적용
## 2) Inference Strategy
- 예측 시에는 이미지의 가로와 세로 중 작은 쪽이 input image size와 같아지도록 resize한 후 center crop했습니다.

# 2. Architecture
- [2]의 구조에서 image classification task에 맞게 일부를 변형했습니다.
    - 마지막 convolutional layer의 출력 차원을 2에서 4로 변경했습니다. (`nn.Conv2d(16, 4, kernel_size=1)`)
    - 마지막 convolutional layer 다음에 global average pooling을 사용해서 spatial dimension을 제거했습니다.
- 전체 학습 가능한 parameters의 수는 약 2,200만 개입니다.
- 변경되지 않은 레이어들에 대해서는 [2]의 공식 저장소에 공개되어있는 parameters를 사용해 fine-tune했습니다.

# 3. Training Details
- Number of epochs: 6
- Input image size: 384

# 4. Performance
- Accuracy 0.922 on test set

# 5. Future Improvements
- 아랍어 외의 다양한 언어에 대해 모델을 학습시킵니다.
- 'Place Translation' 등 다른 서비스에도 사용해서 프로세스를 개선합니다.

# 6. References
- [1] [Arabic Scene Text Recognition in the Deep Learning Era: Analysis on A Novel Dataset](https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text?tab=readme-ov-file)
- [2] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [3] [A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/KimRass/SimCLR/blob/main/papers/a_simple_framework_for_contrastive_learning_of_visual_representations.pdf)
