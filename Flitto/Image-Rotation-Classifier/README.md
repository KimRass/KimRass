- 아랍어를 주로 포함하고 있는 scene text image에 대해서 이미지의 회전된 각도를 0°, 90°, 180°, 270° 중 하나로 분류하는 모델입니다.

# 1. Data
- 자체 크라우드 소싱 데이터 수집 플랫폼을 통해 수집한 아랍어 scene text images 31,652장
- EvArEST dataset [1] (detection) 1,020장
- Trained for 3 epochs
## 1) Data Augmentation
- [3]에서 사용된 data augmentation 기법들을 참고하여 다소 강하게 구현했습니다.
    - 무작위 crop, resize, 가로와 세로의 비율 변환
    - 일정 확률로 좌우 반전
    - 무작위 color jittering
    - 일정 확률로 grayscale로 변환
    - 일정 확률로 무작위 Gaussian blur 적용

# 2. Architecture
- [2]의 구조에서 image classification task에 맞게 일부를 변형했습니다.
    - 마지막 convolutional layer의 출력 차원을 2에서 4로 변경했습니다. (`nn.Conv2d(16, 4, kernel_size=1)`)
    - 마지막 convolutional layer 다음에 global average pooling을 사용해서 spatial dimension을 제거했습니다.
- 전체 학습 가능한 parameters의 수는 약 2,200만 개입니다.
- 변경되지 않은 레이어들에 대해서는 [2]의 공식 저장소에 공개되어있는 parameters를 사용해 fine-tune했습니다.

# 3. Performance
- Accuracy 0.922 on test set

# 4. Future Improvements
- 아랍어 외의 다양한 언어로 확장하여 'Place Translation' 등의 서비스에도 사용함으로써 프로세스를 개선할 필요가 있습니다.

# 5. References
- [1] [Arabic Scene Text Recognition in the Deep Learning Era: Analysis on A Novel Dataset]()
- [2] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [3] [A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/KimRass/SimCLR/blob/main/papers/a_simple_framework_for_contrastive_learning_of_visual_representations.pdf)
