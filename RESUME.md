# 1. Profile
- **Name**: 김종범 (Jongbeom Kim)
- **Birthdate**: 1990.06.18
- **Mobile**: +82-10-4442-6684
- **E-mail**: purflow64@gmail.com
- [**Personal GitHub Repository**](https://github.com/KimRass):
    - AI 교육 자료: 수강한 AI 관련 교육 내용 및 실습 자료.
    - AI 논문 정리 및 분석: 주요 AI 논문들을 주제별로 분류하고, 중요한 내용을 하이라이팅한 자료.
    - 논문 구현: AI 논문을 밑바닥부터 구현한 코드와 학습된 모델을 공유, 실험 결과와 느낀 점을 정리한 자료.

# 2. Core Competencies

## 1) 딥 러닝 논문 구현 경험.
|모델명|내용|
|-|-|
|'ViViT'|- 'Spatio-temporal attention', 'Factorised encoder', 'Factorised self-attention'의 3가지 architecture 구현.|
|'MAE'|- pre-training을 위한 architecture 구현.|
|'Copy-Paste'|- large scale jittering, Copy-Paste 구현<br>- COCO 2014 데이터셋에 적용.|
|'ILVR'|- 'scale factor' 또는 'conditioning range' 변화에 따른 시각화.|
|'SD-Edit'|- 'human-stroke-simulation algorithm' 및 이를 적용한 image editing 구현.|
|'DDPM'|- 32 × 32, 64 × 64의 CelebA에 대해 학습.<br>- denoising process를 gif 파일로 시각화.<br>- linear interpolation, 'coarse-to-fine interpolation'을 통한 sampling 구현.|
|'DDIM'|- spherical linear interpolation, 'grid interpolation'을 통한 sampling 구현.|
|Classifier guidance|- CIFAR-10에 대해 학습.|
|VAE|- MNIST에 대해 학습.<br>- encoder output, decoder 각각에 대한 시각화.|
|'VQ-VAE' & 'PixelCNN'|- Fashion MNIST, CIFAR-10에 대해 학습.<br>- 'PixelCNN'을 통한 sampling.|
|'DCGAN'|- 64 × 64의 CelebA에 대해 학습.<br>- latent space interpolation 구현.|
|'Pix2Pix'|- 'Facades', Google Maps 데이터셋에 대해 학습.|
|'CycleGAN'|- 논문에서 사용한 데이터셋 중 5가지에 대해 학습.|
|'PGGAN'|- 512 × 512의 CelebA-HQ에 대해 학습.|
|'DeepLabv3'|- VOC2012 데이터셋에 대해 학습, 모델 출력 시각화.|
|Gatys et al., 2016 (image style transfer)|- VGGNet-19 기반으로 구현.|
|'ViT'|- 'CutOut', 'Hide-and-Seek', 'CutMix' data augmentation 구현 및 실험.<br>- 'Attention roll-out'을 통한 attention 시각화.<br>- position embedding similarity, position embedding interpolation 구현.<br>- CIFAR-10, CIFAR-100에 대해 학습.|
|'BERT'|- 'BookCorpus', 'SQuAD', 'SWAG' 데이터셋 전처리 구현.<br>- architecture, masked language modeling 구현.|
|'Sentence-BERT'|- classification loss, regression loss, constrastive loss 구현<br>-'STSb', 'WikiSection' 데이터셋 전처리|
|'CLIP'|- 'Flickr8k' + 'Flickr30k'에 대해 학습.<br>- zero-shot classification, linear classification 구현.|
|'CAM'|- pre-trained GoogleNet에 적용.<br>- 모델 출력을 bounding box로 변환.|
|Zhang et al., 2016	(image colorization)|- empirical probability distribution 시각화.|
|'RotNet'|- attention map 시각화.|
|'SimCLR'|- normalized temperature-scaled cross entropy loss, pixel intensity histogram 구현|
|'STEFANN'|- Google Fonts 데이터셋<br>- 'FANnet' architecture 구현 및 Google Fonts 데이터셋에 대해 학습.<br>- Average SSIM.|
| 'PixelLink'|- architecture, instance-balanced cross entropy loss, 모델 출력 후처리 구현|
<!-- - **Vision**: 'VQ-VAE', 'Copy-Paste', 'ILVR', 'Pix2Pix', 'CycleGAN', 'PGGAN', 'DDPM', 'ViT', 'ViViT', 'PixelCNN', 'DeepLabv3', Gatys et al. (2016), etc.
- **Language**: 'BERT', 'Sentence-BERT', 'RoBERTa', etc.
- **Vision-language**: 'CLIP', etc. -->

## 2) Vision 모델 개발 경험.
- scene text image QC 모델.
- image rotation classification 모델.
- textual attribute recognition 모델.
- scene text removal 모델.

## 3) [**알고리즘과 자료구조에 대한 이해.**](https://github.com/KimRass/Algorithm-Coding-Test)
- 프로그래머스 90 문제 해결; Baekjoon Online Judge 153 문제 해결, Gold III 등급.

## 4) Tableau 기반 전사 대시보드 구축 리드 경험 (2021.02 ~ 2022.04).

## 5) 데이터 분석 프로젝트 리드 경험 (2020.01 ~ 2020.09).

# 3. Education
|기간|내용|
|-|-|
|2010.03 ~ 2016.08|연세대학교 신촌 캠퍼스 기계공학과 졸업.|
|2006.03 ~ 2009.02|충북고등학교 졸업.|

# 4. Work Experience
|기간|회사|
|-|-|
|2024.05 ~ 2024.08|Meraker (정규직)|
|2024.03 ~ 2024.05|D-Meta (정규직)|
|2022.05 ~ 2023.12|Flitto (정규직)|
|2016.07 ~ 2022.04|HDC현대산업개발 (정규직)|

## 1) Meraker (2024.05 ~ 2023.08, 정규직)
- 회사 소개: 2022년 설립된 AI 스타트업.
- 퇴직 사유: 경영상 어려움으로 인한 권고사직.

### (1) video super-resolution 모델 연구개발.
- 'BasicVSR++' 모델을 기반으로 Stable Diffusion 3의 pre-trained VAE를 결합하여 latent space에서 계산 효율성 향상.
- 이미지 데이터를 사전에 crop 및 VAE-encode하여 반복 계산을 제거 및 학습 속도 최적화.
- 8대의 RTX 4090과 `accelerate` 라이브러리를 활용하여 데이터 분산 학습 진행.
- TIPS (Tech Incubator Program for Startup) 프로그램 발표 자료의 기술 파트 작성.

## 2) D-Meta (2024.03 ~ 2024.05, 정규직)
- 회사 소개: 철도 신호 업계 중견기업 대아티아이의 자회사로서 AI 기술을 활용한 철도 관제 시스템 개발.
- 이직 사유: AI 연구 업무를 하고자 함.

### (1) 지능형 CCTV 성능 시험인증 취득.
- 한국인터넷진흥원의 '지능형 CCTV 성능 시험 인증'에서 '쓰러짐' 분야 인증 취득.
- 쓰러진 사람을 탐지하는 object detection 모델 개발.
- 실시간으로 영상을 처리하여 이벤트 발생 시각을 추정하는 알고리즘 개발.

### (2) object detection 모델 개발.
- 경남 무인 도시철도 역사에서 유모차, (전동) 휠체어 등의 객체와 실신 등의 상황을 탐지하기 위한 객체 탐지 모델.
- 유료 데이터와 'AI-Hub' 공공 데이터를 조사하여 탐지 대상 객체가 포함된 데이터셋을 선정, 이를 모델 학습에 적합한 포맷으로 변환하는 코드 작성.
- 실제 지하철 CCTV 영상으로부터 탐지 대상 객체가 포함된 장면을 추출하기 위한 코드 작성:
    - 1차 학습된 모델로 장면을 선별한 후, 이를 사람이 검수할 수 있는 시스템 설계.
    - 검수를 쉽게 할 수 있도록 'PyQt' 라이브러리 기반의 GUI 제공.

## 3) Flitto (2022.05 ~ 2023.12, 정규직)
- 회사 소개:
    - [음식점 메뉴 이미지에 대한 다국어 이미지 번역 서비스 제공.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation)
    - B2B 데이터 판매.
    - 크라우드 데이터 레이블링 플랫폼 운영.
    - 사원 약 150명, 2019년 코스닥 상장.
- 이직 사유: AI 기술을 적극적으로 활용하는 회사에서 근무하고자 함.

### (1) [scene text image QC 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Scene-Text-Image-QC)
- 아랍어 scene text images 수집 프로젝트에서 다양한 알고리즘을 통해 데이터 QC 수행하는 모델.
- image rotation classification 모델을 사용해 회전된 이미지를 정상으로 복원.
- image embedding에 기반하여 기존에 수집된 이미지와 유사한 이미지 제외.
- 'EasyOCR' OCR framework를 사용하여 아랍어 텍스트의 수와 비율이 적정한지 검토.

### (2) [image rotation classification 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Image-Rotation-Classifier)
- scene text image에 대해서 이미지의 회전된 각도를 0°, 90°, 180°, 270° 중 하나로 분류하는 모델.
- data augmentation을 통해 모델의 일반화 성능 확보.
- pre-trained scene text detection 모델의 architecture를 수정하여 fine-tuning 실시.

### (3) digital image processing 알고리즘 개발.
- [**automatic image resizing 알고리즘 개발.**](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Automatic-Image-Resizer):
    - 기존 메뉴 이미지를 조금 다른 새로운 이미지로 변경할 때, feature matching을 통해 적절한 해상도로 resize.
    - SIFT (Scale-Invariant Feature Transform), RANSAC (RANdom SAmple Consensus) 사용.
    - 도입 효과: 기존 이미지에 대한 bounding box annotation, 번역 등의 작업 결과물을 재활용하여 업무 효율 향상.
- [**menu image generation 알고리즘 개발.**](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Menu-Image-Generator):
    - 기존의 수작업 메뉴 이미지 제작을 Python으로 템플릿화하여 자동화.
    - 도입 효과: 비효율적이고 반복적인 프로세스 개선, 시각적 품질 향상.

### (4) [textual attribute recognition 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Textual-Attribute-Recognizer)
- 메뉴 이미지의 각 텍스트에서 7가지 속성 (font size, writing direction, text alignment, text line breaking, text color, text border, text border color)을 추출하는 모델.
<!-- - 'CRAFT' scene text detection 모델을 통해 추출된 'text region score map'을 기반으로 알고리즘 설계. -->
- 자체 데이터셋을 통해 학습된 semantic segmentation 모델로 text alignment 추출.
- 언어별 특성을 고려하여 font size를 최적화하는 동시에 텍스트에 줄바꿈 삽입 위치를 계산.
- contrast ratio를 기반으로 픽셀 단위의 정교한 계산을 통해 텍스트의 가독성 개선.
- 번역된 텍스트 렌더링 시, 속성을 반영하여 시각적 품질 향상.

### (5) [scene text removal 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Scene-Text-Remover)
- 메뉴 이미지에서 번역이 필요한 텍스트를 자연스럽게 지우는 모델.
- scene text detection → text stroke mask prediction → image inpainting의 3단계로 작동.
- thresholding, connected-component labeling, fully connected CRF 등 다양한 digital image proessing 기법 활용.
- 지우고자 하는 문자의 중심을 감싸도록 박스를 그림으로써 원하는 텍스트만 선택적으로 제거 가능.
- 기존 포토샵을 통한 수작업 대비 처리 속도 개선 및 텍스트가 있는 영역만 수정함으로써 시각적 품질 향상.

### (6) [data QC 알고리즘 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Data-Collection)
- B2B 고객사의 의뢰에 따라 데이터를 수집하여 납품하는 프로젝트에 있어서 자동 데이터 QC를 위한 각종 알고리즘 개발.
- 텍스트 데이터:
    - 단어별 표준 띄어쓰기 규칙을 반영하여 문장 교정 (e.g., '빅데이터' → '빅 데이터').
    - 형태소 분리에 기반하여 존댓말을 반말로 변환.
    - 'Sentence-BERT'를 활용한 문장 유사도 검사.
    - 형태소 분리에 기반하여 완전한 문장과 불완전한 문장 구분 및 완전한 문장으로 이루어진 데이터 수집.
- 음성 데이터:
    - QA 요구조건에 따라 발화 구간의 길이와 수, 음성의 크기, 노이즈 등 검토.
    - 음성 데이터 시각화 (waveform, spectrogram, mel-spectrogram)를 통해 수동 QC 보조.

## 4) HDC현대산업개발 (2016.07 ~ 2022.04, 정규직)
- 회사 소개:
    - 종합건설업 (아파트 브랜드 'IPARK')
    - 사원 약 1,800명, 매출액 약 3조 원
- 이직 사유: AI에 관련된 업무 수행 및 그에 대한 전문성 향상.

###	(1) [Tableau 기반의 전사 대시보드 구축 리드 (2021.02 ~ 2022.04).](https://github.com/KimRass/KimRass/tree/main/HDC/Tableau-BI)
- 현업과 협의하여 대시보드 설계 및 Tableau와 data mart 연결.
- 자동 데이터 업데이트를 위한 Tableau 로직 설계.
- SQL을 통해 data mart와 Tableau 로직 점검.

### (2) [아파트 실거래가 데이터 분석 리드 (2020.01 ~ 2020.09).](https://github.com/KimRass/KimRass/tree/main/HDC/APT-Price-Analysis)
- 아파트 실거래가 데이터 분석을 통해 데이터 기반 적정 분양가 산정.
- web scraping을 통해 '네이버 부동산' 등에서 부동산 관련 데이터 수집.
- feature engineering 및 'XGBoost' 기반 modeling.
- 'Mapbox' 기반 데이터 시각화.
- 최근 10년간 전용면적 또는 층에 대한 선호도 변화 분석.

# 5. Certificates
- SQLD (한국데이터산업진흥원) (2021.12)

# 6. Foreign Languages
- 영어: 논문을 읽고 이해하는 데 문제 없음.
- 일본어: 초중급 정도의 말하기, 읽기, 듣기 가능.

# 7. Training
||||
|-|-|-|
|2022.01 ~ 2022.02|'4가지 유즈 케이스를 활용한 시계열 분석: 전처리부터 딥러닝 적용까지 5기'|'러닝스푼즈'|
|2021.09 ~ 2021.10|'유지·보수 비용 10배 절감을 위한 DB 설계 및 구축 1기'|'러닝스푼즈'|
|2021.07 ~ 2021.08|'태블로 마스터 클래스'|'VizLab'|
|2021.03 ~ 2021.04|'컴퓨터 비전: 딥러닝 기반의 영상 인식 모델 구현 2기'|'러닝스푼즈'|
|2020.09 ~ 2020.12|'TensorFlow를 활용한 딥러닝 자연어처리 입문 6기'|'러닝스푼즈'|
|2020.08 ~ 2020.09|'고객데이터와 딥러닝을 활용한 추천시스템 구현'|'러닝스푼즈'|
