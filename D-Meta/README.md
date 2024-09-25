# 1. KISA
- 침입, 싸움, 쓰러짐, 

# 2. Object Detection for Surveilance Camera
- Model: YOLOv8
- Target objects: 유모차, (전동) 휠체어, 쓰러짐.

## 1) 데이터 수집

### (1) AI-Hub
- 'CCTV 추적 영상' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=160, 'cctv.py', 'cctv.sh'):
    - 휠체어, 유모차.
    - 이미지.
- '인도보행 영상' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189, 'street.py', 'street.sh'):
    - 자전거, 휠체어.
    - 이미지.
- '시니어 이상행동 영상' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=167, senior.py)
- '이상행동 CCTV 영상' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=171, 'abnormal.py')

### (2) 부산 명륜역 CCTV 영상
- 최근 1년간의 명륜역사 CCTV 영상으로부터 목표 오브젝트에 대한 데이터 확보.
