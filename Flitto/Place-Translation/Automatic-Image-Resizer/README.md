# <이미지 크기 자동 조정 알고리즘 개발>

# 1. 문제 정의
- 기존에 접수된 이미지에서, 메뉴 구성이나 항목의 가격 변화가 발생하는 경우가 있습니다. 새로 접수된 이미지는 기존 이미지와 약간의 차이점만 존재하므로 처음부터 모든 작업을 하기보다 기존의 작업물을 재활용하는 것이 바람직합니다.
- 이를 위해서 기존에는 디자인팀이 포토샵을 통해 수작업으로 기존 이미지를 바탕으로 새로운 이미지의 해상도를 수정해왔습니다. 이 모습을 가만히 지켜보다가 이미지 처리 알고리즘으로 자동화할 수 있겠다는 생각이 들어 이 기능을 개발하게 되었습니다.

# 2. 문제 해결
- 기존 이미지와 새로운 이미지 간의 feature matching을 통해 새로운 이미지를 적절한 해상도로 자동 조정하는 알고리즘 개발.
- SIFT (Scale-Invariant Feature Transform)를 활용해 두 이미지 간의 특징점 추출 후, RANSAC (RANdom SAmple Consensus)을 사용해 최적의 호모그래피 행렬을 계산합니다.
- 새로운 이미지의 해상도를 조정한 후 빈 영역이 생기면 기존 이미지로부터 그대로 가져와 채웁니다. 반대로 기존 이미지를 벗어나는 영역이 생기면 잘라냅니다.
- RabbitMQ를 기반으로 비동기 AMQP (메세지 큐) 서버를 개발하고 백 오피스와 연동했습니다.
- 사용자가 새로운 이미지를 PDF 파일로 등록할 경우, feature matching이 가장 많이 발생한 페이지를 자동으로 찾는 기능을 추가로 구현했습니다.

# 3. 성과
- 기존 수작업 대비 이미지 번역의 처리 속도를 대폭 개선했습니다.

## 1) 예시
- 음식의 가격이 바뀐 경우
   |기존 이미지 (2,195 × 2,252)|새로운 이미지 (9,669 × 9,670)|해상도 조정 (2,195 × 2,252)|
   |-|-|-|
   |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Automatic-Image-Resizer/examples/eataly_old.jpg" width="300">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Automatic-Image-Resizer/examples/eataly_new.jpg" width="500">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Automatic-Image-Resizer/examples/eataly_new_resized.png" width="300">|
- 음식 이름이 바뀐 경우
   |기존 이미지 (1,240 × 1,754)|새로운 이미지 (1,654 × 2,339)|해상도 조정 (1,240 × 1,754)|
   |-|-|-|
   |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Automatic-Image-Resizer/examples/langman1_old.jpg" width="300">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/portfolio/resources/langman-new_image.jpg" width="400">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Automatic-Image-Resizer/examples/langman1_new_resized.png" width="300">|
