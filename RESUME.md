# 1. Profile
- **Name**: 김종범 (Jongbeom Kim)
- **Birthdate**: 1990.06.18
- **Mobile**: 010-4442-6684
- **E-mail**: purflow64@gmail.com
- **GitHub**: [KimRass](https://github.com/KimRass)

# 2. Core Competencies
- **딥 러닝 논문을 PyTorch로 From Scratch 구현 경험**:
    - **Vision**: VAE, DCGAN, Pix2Pix, CycleGAN, PGGAN, DDPM, ViT, DeepLab v3, Gatys et al. (2016), 등.
    - **Language**: BERT, Sentence-BERT, RoBERTa, 등.
    - **Vision-language**: CLIP, 등.
- **Vision 모델 개발 경험**:
    - Scene text image QC 모델
    - Image rotation classification 모델
    - Textual attribute recognition 모델
    - Scene text removal 모델
- **Tableau 기반 전사 대시보드 구축 리드 경험**
- **아파트 실거래가 데이터 분석 리드 경험**
- **[알고리즘과 자료구조에 대한 이해](https://github.com/KimRass/Algorithm-Coding-Test)**:
    - 프로그래머스 90 문제 해결; Baekjoon Online Judge 153 문제 해결, Gold III 등급.

# 3. Education
| 기간 | 내용 |
|-|-|
| 2010.03 ~ 2016.08 | 연세대학교 신촌 캠퍼스 기계공학과 졸업. |
| 2006.03 ~ 2009.02 | 충북고등학교 졸업. |

# 4. Work Experience
| 기간 | 회사 |
|-|-|
| 2024.05 ~ 2024.08 | Meraker (정규직) |
| 2024.03 ~ 2024.05 | D-Meta (정규직) |
| 2022.05 ~ 2023.12 | Flitto (정규직) |
| 2016.07 ~ 2022.04 | HDC현대산업개발 (정규직) |

## 1) Meraker (2024.05 ~ 2023.08, 정규직)
- 회사 소개: 2022년 설립된 AI 스타트업.
- 퇴직 사유: 경영상 어려움으로 인한 권고사직.

### (1) Video super-resolution 모델 연구개발.
- 'BasicVSR++' 모델을 기반으로 Stable Diffusion 3의 Pre-trained VAE를 결합하여 Latent space에서 계산 효율성 향상.
- 이미지 데이터를 사전에 Crop 및 VAE-encode하여 반복 계산을 제거하고 학습 속도 최적화.
- 8대의 RTX 4090과 `accelerate` 라이브러리를 활용하여 데이터 분산 학습 진행.
- TIPS (Tech Incubator Program for Startup) 프로그램 발표 자료의 기술 파트 작성.

## 2) D-Meta (2024.03 ~ 2024.05, 정규직)
- 회사 소개: 철도 신호 업계 중견기업 대아티아이의 자회사로서 AI 기술을 활용한 철도 관제 시스템 개발.
- 이직 사유: AI 연구 업무를 하고자 함.

### (1) 지능형 CCTV 성능 시험인증 취득.
- 한국인터넷진흥원의 '지능형 CCTV 성능 시험 인증'에서 '쓰러짐' 분야 인증 취득.
- 쓰러진 사람을 탐지하는 Object detection 모델 개발.
- 실시간으로 영상을 처리하여 이벤트 발생 시각을 추정하는 알고리즘 개발.

### (2) Object detection 모델 개발.
- 경남 무인 도시철도 역사에서 유모차, (전동) 휠체어 등의 객체와 실신 등의 상황을 탐지하기 위한 객체 탐지 모델.
- 유료 데이터와 'AI-Hub' 공공 데이터를 조사하여 탐지 대상 객체가 포함된 데이터셋을 선정하고, 이를 모델 학습에 적합한 포맷으로 변환하는 코드 작성.
- 실제 지하철 CCTV 영상으로부터 탐지 대상 객체가 포함된 장면을 추출하기 위한 코드 작성:
    - 1차 학습된 모델로 장면을 선별한 후, 이를 사람이 검수할 수 있는 시스템 설계.
    - 검수를 쉽게 할 수 있도록 PyQt 라이브러리 기반의 GUI 제공.

## 3) Flitto (2022.05 ~ 2023.12, 정규직)
- 회사 소개:
    - [음식점 메뉴 이미지에 대한 다국어 이미지 번역 서비스 제공.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation)
    - B2B 데이터 판매.
    - 크라우드 데이터 레이블링 플랫폼 운영.
    - 사원 약 150명, 2019년 코스닥 상장.
- 이직 사유: AI 기술을 적극적으로 활용하는 회사에서 근무하고자 함.

### (1) [Scene text image QC 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Scene-Text-Image-QC)
- 아랍어 Scene text images 수집 프로젝트에서 다양한 알고리즘을 통해 데이터 QC 수행하는 모델.
- Image rotation classification 모델을 사용해 회전된 이미지를 정상으로 복원.
- Image embedding에 기반하여 기존에 수집된 이미지와 유사한 이미지 제외.
- 'EasyOCR' OCR framework를 사용하여 아랍어 텍스트의 수와 비율이 적정한지 검토.

### (2) [Image rotation classification 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Image-Rotation-Classifier)
- Scene text image에 대해서 이미지의 회전된 각도를 0°, 90°, 180°, 270° 중 하나로 분류하는 모델.
- Data augmentation을 통해 모델의 일반화 성능 확보.
- Pre-trained scene text detection 모델의 Architecture를 수정하여 Fine-tuning 실시.

### (3) Image processing 알고리즘 개발.
- [Automatic image resizing 알고리즘 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Automatic-Image-Resizer)
    - 기존 메뉴 이미지를 조금 다른 새로운 이미지로 변경할 때, Feature matching을 통해 적절한 해상도로 resize.
    - SIFT (Scale-Invariant Feature Transform)와 RANSAC (RANdom SAmple Consensus) 사용.
    - 도입 효과: 기존 이미지에 대한 Bounding box annotation, 번역 등의 작업 결과물을 재활용하여 업무 효율 향상.
- [Menu image generation 알고리즘 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Menu-Image-Generator)
    - 기존의 수작업 메뉴 이미지 제작을 Python으로 템플릿화하여 자동화.
    - 도입 효과: 비효율적이고 반복적인 프로세스 개선, 시각적 이미지 번역 품질 향상.

### (4) [Textual attribute recognition 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Textual-Attribute-Recognizer)
- 메뉴 이미지에서 각 텍스트의 속성을 인식하는 모델로서, 이 속성들을 번역문 렌더링 시 활용.
    - 속성: Font size, Writing direction, Text alignment, Text line breaking, Text color, Text border, Text border color
- 도입 효과: 시각적 이미지 번역 품질 향상.

### (5) [Scene text removal 모델 개발.](https://github.com/KimRass/KimRass/tree/main/Flitto/Place-Translation/Scene-Text-Remover)
- 메뉴 이미지에서, 사용자가 지우고자 하는 텍스트만을 자연스럽게 지우는 모델.
- Scene text detection → Text stroke mask prediction → Image inpainting의 3단계로 작동.
- 도입 효과: 수작업 감소, 이미지 처리 속도 향상, 시각적 이미지 번역 품질 향상.

### (6) [Data QC.](https://github.com/KimRass/KimRass/tree/main/Flitto/Data-Collection)
- B2B 고객사의 의뢰에 따라 데이터를 수집하여 납품하는 프로젝트에 있어서 자동 데이터 QC를 위한 각종 알고리즘 개발.
    - 다국어 병렬 말뭉치 (텍스트 데이터)
        - 번역 모델 학습을 위한 다국어 데이터 수집 (ko, en, hi, km, tl, ru, vi, id, th).
    - 한국어 사전 예문 (텍스트 데이터)
        - 전문용어, 공공용어, 신조어로 구분된 단어 및 예문 수집.
        - 단어별 표준 띄어쓰기 규칙을 반영하여 문장 교정 (e.g., '빅데이터' -> '빅 데이터').
    - 명대사를 포함하는 한국어 대화문 (텍스트 데이터)
        - 영화 명대사 및 속담이 사용된 Chatbot 학습 데이터 수집.
        - 형태소 분리에 기반하여 존댓말을 반말로 변환.
        - Sentence-BERT를 활용한 문장 유사도 검사.
    - 한국어와 영어 뉴스 기사 (텍스트 데이터)
        - 형태소 분리에 기반하여 완전한 문장과 불완전한 문장을 구분하고, 완전한 문장으로 이루어진 데이터 수집.
    - 한국어 명령어 발화 (음성 데이터)
        - "선풍기 켜 줘"와 같은 짧은 명령어를 발화한 음성 데이터 수집.
        - QA 요구조건에 따라 발화 구간의 길이와 수, 음성의 크기, 노이즈 등 검토.
        - 음성 데이터 시각화 (Waveform, Spectrogram, Mel-spectrogram).

## 4) HDC현대산업개발 (2016.07 ~ 2022.04, 정규직)
- 회사 소개:
    - 종합건설업 (아파트 브랜드 'IPARK')
    - 사원 약 1,800명, 매출액 약 3조 원
- 이직 사유: AI와 관련된 업무 수행 및 그에 대한 전문성 향상.

###	(1) [Tableau 기반의 전사 대시보드 구축 리드.](https://github.com/KimRass/KimRass/tree/main/HDC/Tableau-BI) (2021.02 ~ 2022.04)
- 현업과 협의하여 대시보드 설계 및 Tableau와 Data mart 연결.
- 자동 데이터 업데이트를 위한 Tableau 로직 설계.
- SQL을 통해 Data mart와 Tableau 로직 점검.

### (2) [아파트 실거래가 데이터 분석 리드.](https://github.com/KimRass/KimRass/tree/main/HDC/APT-Price-Analysis) (2020.01 ~ 2020.09)
- 아파트 실거래가 데이터 분석을 통해 데이터 기반 적정 분양가 산정.
- Web scraping을 통해 '네이버 부동산' 등에서 부동산 관련 데이터 수집.
- Feature engineering 및 'XGBoost' 기반 modeling.
- 'Mapbox' 기반 데이터 시각화.
- 최근 10년간 전용면적 또는 층에 대한 선호도 변화 분석.

# 5. Certificates
- SQLD (한국데이터산업진흥원) (2021.12)

# 6. Foreign Languages
- 영어: 논문을 읽고 이해하는 데 문제 없음.
- 일본어: 초중급 정도의 말하기, 읽기, 듣기 가능.

# 7. Training
| | | |
|-|-|-|
| 2022.01 ~ 2022.02 | '4가지 유즈 케이스를 활용한 시계열 분석: 전처리부터 딥러닝 적용까지 5기' | '러닝스푼즈' |
| 2021.09 ~ 2021.10 | '유지·보수 비용 10배 절감을 위한 DB 설계 및 구축 1기' | '러닝스푼즈' |
| 2021.07 ~ 2021.08 | '태블로 마스터 클래스' | 'VizLab' |
| 2021.03 ~ 2021.04 | '컴퓨터 비전: 딥러닝 기반의 영상 인식 모델 구현 2기' | '러닝스푼즈' |
| 2020.09 ~ 2020.12 | 'TensorFlow를 활용한 딥러닝 자연어처리 입문 6기' | '러닝스푼즈' |
| 2020.08 ~ 2020.09 | '고객데이터와 딥러닝을 활용한 추천시스템 구현' | '러닝스푼즈' |
