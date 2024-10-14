- 데이터: 자체 수집한 이미지 31,652장 및 EvArEST 데이터셋 510장 사용.
- 모델 아키텍처: 사전 학습된 텍스트 탐지 모델을 분류 태스크에 맞게 일부 수정.
- 테스트셋에 대한 정확도 88.2% 기록.

# 1. 문제 정의
- 아랍어 이미지 수집 프로젝트에서 사용자들이 가끔 회전된 이미지를 업로드하는 경우가 있었습니다. 따라서 이미지의 회전 각도를 0°, 90°, 180°, 270° 중 하나로 분류하는 모델을 개발하여 자동으로 이미지가 복원되도록 시도했습니다.

# 2. 문제 해결

## 1) Data
- 자체 크라우드 소싱 데이터 수집 플랫폼을 통해 수집한 이미지 (31,652장) +  EvArEST [1] detection dataset (510장)을 사용했습니다.
- [3]을 참고하여 data augmentation을 도입했으며, 추가적으로 무작위의 작은 각도로 이미지를 회전시켜 본 프로젝트에서 실제로 수집되는 이미지와 비슷한 데이터를 증강하고자 했습니다.

## 2) 아키텍처
- 사전 학습된 'CRAFT' [2]의 파라미터를 가져오고 분류 태스크에 맞게 아키텍처를 일부 수정했습니다. global average pooling과 FC 레이어를 사용하여 4개의 클래스 각각의 확률을 출력하도록 했습니다.

## 3) 훈련
- 기존 모델의 레이어와 새로 추가된 레이어 대해서 서로 다른 learning rate를 적용했습니다.
- 어떤 각도를 가지고 이미지를 직접 회전시키지 않고 flip과 transpose를 통해 이미지의 회전을 구현했습니다 [4].
- minibatch를 구성할 때, 하나의 이미지에 대해서 무작위로 4가지 각도 중 하나로 회전시키지 않고 4가지 각도에 대해서 각각 회전시킨 것을 전부 포함시켰습니다 [4].
<!-- - Number of epochs: 6
- Input image size: 384 -->

# 3. 성과
## 1) 성능
- Accuracy on test set: 0.882

# 4. 향후 개선점
- 아랍어 외의 다양한 언어에 대해 모델을 학습시킵니다.
- 학습된 모델을 'Place Translation' 등 다른 서비스에도 적용해서 이미지가 잘못 회전되어 들어오는 경우에 대한 프로세스를 개선합니다.
- scene text가 매우 작은 경우에 모델의 성능이 상대적으로 떨어지므로 이를 개선합니다.
- scene text 외의 시각적인 요소만 가지고는 이미지의 회전 각도를 추론하기 어렵고 scene text의 회전 각도를 주요하게 봐야만 하는 이미지에 대해서는 모델의 성능이 매우 낮게 나타나는 것으로 파악되어 이를 개선합니다.

# 5. References
- [1] [Arabic Scene Text Recognition in the Deep Learning Era: Analysis on A Novel Dataset](https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text)
- [2] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [3] [A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/KimRass/SimCLR/blob/main/papers/a_simple_framework_for_contrastive_learning_of_visual_representations.pdf)
- [4] [Unsupervised Representation Learning by Predicting Image Rotations](https://github.com/KimRass/RotNet/blob/main/papers/RotNet/unsupervised_representation_learning_by_predicting_image_rotations.pdf)




