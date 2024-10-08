- 아랍어를 주로 포함하고 있는 scene text image에 대해서 이미지의 회전된 각도를 0°, 90°, 180°, 270° 중 하나로 분류하는 모델입니다.

# 1. Data
- 자체 크라우드 소싱 데이터 수집 플랫폼을 통해 수집한 이미지 (31,652장) +  EvArEST [1] detection dataset (510장)
- 전체 데이터의 각각 80%, 10%, 10%을 Training set, Valiation set, Test set으로 나눴습니다.
## 1) Data Augmentation
- [3]에서 사용된 data augmentation 기법들을 참고하여 다소 강하게 구현했습니다.
    - (작은 각도로) image rotation
    - image cropping
    - color jittering (hue, saturation, brightness, contrast) 적용
    - Gaussian blur 적용
## 2) Inference Strategy
- 예측 시에는 이미지의 가로와 세로 중 작은 쪽이 input image size와 같아지도록 resize한 후 center crop했습니다.

# 2. Architecture
- [2]의 구조에서 분류 태스크에 맞게 일부를 변형했습니다.
    <!-- - 마지막 Convolutional layer의 출력 차원을 2에서 4로 변경했습니다. (`nn.Conv2d(16, 4, kernel_size=1)`) -->
    - 마지막 convolutional layer 다음에 global average pooling을 사용해서 spatial dimension을 제거했습니다.
    - 이후에 두 개의 FC layer를 사용해서 4개의 클래스 대한 확률을 출력하도록 했습니다.
- 전체 학습 가능한 파라미터의 수는 약 2,000만 개입니다.
- 변경되지 않은 레이어들에 대해서는 [2]의 공식 저장소에 공개되어있는 Parameters를 사용해 Fine-tune했습니다.

# 3. Training Details
- 기존 [2]의 레이어들에 대해서는 비교적 작은 learning rate를, 새로 추가된 레이어들에 대해서는 비교적 큰 learning rate를 적용했습니다.
- 어떤 각도를 가지고 이미지를 직접 회전시키지 않고 flip과 transpose를 통해 이미지의 회전을 구현했습니다 [4].
- minibatch를 구성할 때, 하나의 이미지에 대해서 무작위로 4가지 각도 중 하나로 회전시키지 않고 4가지 각도에 대해서 각각 회전시킨 것을 전부 포함시켰습니다 [4].
<!-- - Number of epochs: 6
- Input image size: 384 -->

# 3. Performance
- Accuracy on test set: 0.882

# 4. Future Improvements
- 아랍어 외의 다양한 언어에 대해 모델을 학습시킵니다.
- 학습된 모델을 'Place Translation' 등 다른 서비스에도 적용해서 이미지가 잘못 회전되어 들어오는 경우에 대한 프로세스를 개선합니다.
- scene text가 매우 작은 경우에 모델의 성능이 상대적으로 떨어지므로 이를 개선합니다.
- scene text 외의 시각적인 요소만 가지고는 이미지의 회전 각도를 추론하기 어렵고 scene text의 회전 각도를 주요하게 봐야만 하는 이미지에 대해서는 모델의 성능이 매우 낮게 나타나는 것으로 파악되어 이를 개선합니다.

# 5. References
- [1] [Arabic Scene Text Recognition in the Deep Learning Era: Analysis on A Novel Dataset](https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text?tab=readme-ov-file)
- [2] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [3] [A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/KimRass/SimCLR/blob/main/papers/a_simple_framework_for_contrastive_learning_of_visual_representations.pdf)
- [4] [Unsupervised Representation Learning by Predicting Image Rotations](https://github.com/KimRass/RotNet/blob/main/papers/RotNet/unsupervised_representation_learning_by_predicting_image_rotations.pdf)
