# 1. Profile
- Name: 김종범 (Jongbeom Kim)
- Birthdate: 1990.06.18
- Mobile: 010-4442-6684
- E-mail: purflow64@gmail.com
- Github: https://github.com/KimRass

# 2. Core Competence
- [Tableau 기반의 전사 대시보드 구축 리드](#1-tableau-기반의-전사-대시보드-구축-리드-202102--202204)
- [아파트 실거래가 데이터 분석 리드](#2-아파트-실거래가-데이터-분석-리드-202001--202011)
- vision model 개발
    - [scene text removal model](#scene-text-removal-model-개발-202209--202301)
    - [textual attribute recognition model](#textual-attribute-recognition-model-개발-202301--202306)
    - [scene text image rotation classification model](#scene-text-image-rotation-classification-model-개발-202300--202300)
- language model 개발
    - [language identification model](#language-identification-model-개발-202300--202300)
    - [multilingual text embedding model](#multilingual-text-embedding-model-개발-202300--202300)
- image processing
    - ㅇ
- 알고리즘과 자료구조에 대한 이해 ('Baekjoon Online Judge' Gold III 등급)
- 딥 러닝에 대한 깊은 이해 및 PyTorch 기반의 from scratch 구현 경험
    - Vision: DCGAN, Pix2Pix, CycleGAN, PGGAN, DDPM, ViT, DeepLab v3, Gatys et al. (2016), etc.
    - Language: BERT, SentenceBERT, RoBERTa, ALBERT, etc.
    - Vision-language: CLIP, etc.

# 3. Education
| | |
|-|-|
| 2010.03 ~ 2016.08 | 연세대학교 신촌 캠퍼스 기계공학과 졸업 |
| 2006.03 ~ 2009.02 | 충북고등학교 졸업 |

# 4. Work Experience
| | |
|-|-|
| 2022.05 ~ 2023.12 | ['Flitto' (정규직)](#1-flitto-202205--202312-정규직) |
| 2016.07 ~ 2022.04 | ['HDC현대산업개발' (정규직)](#2-hdc현대산업개발-201607--202204-정규직) |
## (1) 'Flitto' (2022.05 ~ 2023.12, 정규직)
- 퇴직 사유:
- 회사 소개:
    - 번역 플랫폼과 크라우드 데이터 레이블링 플랫폼 운영
    - B2B 데이터 판매
    - 사원수 약 150명, 2019년 코스닥 상장
- Data dev. team & Place lab 소속
- 음식점의 메뉴 이미지에 대한 다국어 이미지 번역 서비스 '플리토 플레이스' ('Flitto Place')와 관련된 image processing 알고리즘 개발.
### 1) image processing
- automatic image resizing
    - 기존 메뉴 이미지에서 가격 등 약간의 변화만 있는 새로운 이미지로의 변경이 필요할 때, feature matching을 통해 자동으로 새로운 이미지를 적절한 해상도로 resize.
    - SIFT (Scale-Invariant Feature Transform)와 RANSAC (RANdom SAmple Consensus) 사용.
- menu image merge
    - 하나의 음식점의 메뉴 이미지가 매우 여러 장일 때 상하 또는 좌우로 2장씩 합침으로써 사용자 경험 향상.
    - 이미지의 테두리 색상 결정,
- menu image generation
    - 메뉴 이미지를 직접 제작해 주는 경우, 기존의 Figma를 통한 수작업이 아니라 언어별 번역문 데이터를 바탕으로 python을 통해 자동으로 생성.
    - Figma를 통해 수동으로 메뉴 이미지를 제작하는 수고 절감.
    - 
### 2) scene text image QC model 개발 (2023.00 ~ 2023.00)
- Ref: https://github.com/KimRass/Flitto-ML/tree/main/Scene-Text-Image-QC
### 3) scene text image rotation classification model 개발 (2023.00 ~ 2023.00)
- Ref: https://github.com/KimRass/Flitto-ML/tree/main/Scene-Text-Image-QC/Image-Rotation-Classifier
### 4) language identification model 개발 (2023.00 ~ 2023.00)
- Ref: https://github.com/KimRass/Flitto-ML/tree/main/Language-Identifier
### 5) multilingual text embedding model 개발 (2023.00 ~ 2023.00)
- Ref: https://github.com/KimRass/Flitto-ML/tree/main/Multilingual-Text-Embedder
### 6) textual attribute recognition model 개발 (2023.01 ~ 2023.06)
- Ref: https://github.com/KimRass/Flitto-ML/tree/main/Place-Translation/Textual-Attribute-Recognizer
- 원본 이미지의 각 텍스트 객체로부터 그 속성을 추출하고 이를 각 텍스트의 번역문 렌더링 시 활용.
- 도입 효과: 시각적인 이미지 번역 품질 향상.
- 함으로써 이미지 번역의 가독성과 심미성을 향상시키기 위한 모델.
<!-- - 텍스트 속성은 다음의 7가지를 의미함; font size, writing direction, text alignment, text line breaking, text color, text border color, text border width -->
<!-- - Text region recognition은 텍스트가 렌더링될 수 있는 영역을 새롭게 계산하는 기술입니다. 이를 통해 한국어를 영어로 번역할 때와 같이 원문보다 번역문이 긴 경우에도 원본에 가까운 Font size로 텍스트를 렌더링할 수 있게 됩니다. -->
### 7) Scene text removal model 개발 (2022.09 ~ 2023.01)
- Ref: https://github.com/KimRass/Flitto-ML/tree/main/Place-Translation/Scene-Text-Remover
- 메뉴 이미지에 존재하는 텍스트를 자연스럽게 지우는 모델
- 텍스트간에 서로 가깝게 붙어있는 상황에서 원하는 텍스트만 선별해서 지우는 것이 가능.
- 크게 3단계로 작동 (scene text detection → text stroke mask prediction → image inpainting).
- 도입 효과: 수작업 감소, 이미지 처리 속도 향상, 시각적인 이미지 번역 품질 향상.
### 8) 데이터셋 구축 (2022.05 ~ 2022.11)
- Ref: https://github.com/KimRass/Data-Collection
- B2B 고객사의 의뢰에 따라 데이터셋을 구축하는 프로젝트에 있어서 최소한의 수작업으로 데이터 품질을 높이기 위한 각종 알고리즘을 개발.
    - 다국어 병렬 말뭉치 (텍스트 데이터)
        - machine translation model 학습을 위한 다국어 데이터셋 구축 (ko, en, hi, km, tl, ru, vi, id, th).
    - 한국어 사전 예문 (텍스트 데이터)
        - 전문용어, 공공용어, 신조어로 구분된 각 단어와 예문으로 구성된 데이터셋 구축.
        - 단어별 표준 띄어쓰기 규칙을 반영하여 문장 교정 (e.g., '빅데이터' -> '빅 데이터').
    - 명대사를 포함하는 한국어 대화문 (텍스트 데이터)
        - 영화 명대사나 속담 등을 활용하여 문장을 생성하는 chatbot model을 학습시키기 위한 데이터셋 구축.
        - 형태소 분리에 기반하여 존댓말을 반말로 변환.
        - SentenceBERT에 기반한 문장간 유사도 검사.
    - 한국어와 영어 뉴스 기사 (텍스트 데이터)
        - 형태소 분리에 기반하여 완전한 문장과 불완전한 문장을 구분하고 완전한 문장으로 이루어진 데이터셋 구축.
    - 한국어 명령어 발화 (음성 데이터)
        - "선풍기 켜 줘"와 같은 짧은 명령어를 발화한 음성 데이터셋 구축.
        - 요구조건에 따른 QA (e.g., 발화 구간의 길이와 수, 음성의 크기, 노이즈 등).
        - 음성 데이터 시각화 (waveform, spectrogram, mel-spectrogram).
## (2) 'HDC현대산업개발' (2016.07 ~ 2022.04, 정규직)
- 이직 사유: 인공지능에 대한 전문성 향상
###	1) Tableau 기반의 전사 대시보드 구축 리드 (2021.02 ~ 2022.04)
- Ref: https://github.com/KimRass/Tableau/blob/main/eis.md
- 현업과 협의하여 대시보드 설계 및 Tableau와 data mart 연결.
- 자동으로 대시보드의 데이터가 업데이트될 수 있도록 Tableau 로직 설계 
- SQL을 통해 data mart와 Tableau 로직 점검.
### 2) 아파트 실거래가 데이터 분석 리드 (2020.01 ~ 2020.11)
- Ref: https://github.com/KimRass/KimRass/blob/main/apartments_sales_price_analytics.md
- 아파트 실거래가 데이터 분석을 통해 데이터 기반의 적정 분양가 산정 알고리즘의 가능성 탐색 목적.
<!-- - 청주시 아파트 실거래가에 대해 2020.06 ~ 2021.05 발생한 데이터로 학습을 시키고 2021.06 발생한 데이터로 모델을 평가했습니다. -->
<!-- - 공시지가, 전용면적, 반경 내 학교의 수 등을 Feature로 사용했습니다. 모델은 XGBoost를 사용했으며 Loss function으로는 MSE를, Evaluation metric으로는 MAPE를 사용했습니다. -->
- 최근 10년간 전용면적 또는 층에 대한 선호도 변화 분석 데이터 마이닝 수행.
### 3) '김포 사우 IPARK' 아파트 기계설비 현장 관리 (2016.07 ~ 2018.04)
- 신축 아파트 공사 현장의 기계설비 부문 현장 관리.
- 터파기 공정 시 투입되어 준공 후 한 달 간의 하자보수 기간까지 근무.

# 5. Certificates
- SQLD (한국데이터산업진흥원) (2021.12)

# 6. Foreign Languages
- 영어: 논문을 읽고 이해하는 데 문제 없음.
- 일본어: 초중급 정도의 말하기, 읽기, 듣기 가능.

# 7. Training
- '4가지 유즈 케이스를 활용한 시계열 분석: 전처리부터 딥러닝 적용까지 5기' (러닝스푼즈) (2022.01 ~ 2022.02, 18h)
- '유지·보수 비용 10배 절감을 위한 DB 설계 및 구축 1기' (러닝스푼즈) (2021.09 ~ 2021.10, 18h)
- '태블로 마스터 클래스' (VizLab) (2021.07 ~ 2021.08)
- '컴퓨터 비전: 딥러닝 기반의 영상 인식 모델 구현 2기' (러닝스푼즈) (2021.03 ~ 2021.04, 18h)
- 'TensorFlow를 활용한 딥러닝 자연어처리 입문 6기' (러닝스푼즈) (2020.11 ~ 2020.12, 24h)
- '고객데이터와 딥러닝을 활용한 추천시스템 구현' (러닝스푼즈) (2020.08 ~ 2020.09, 18h)

# 8. Annual Salary

# 9. Cover Letter
## (1) Motivation
- 저는 기계공학을 전공하여 국내 굴지의 건설사에서 기계설비 직무로 첫 사회생활을 시작했습니다. 그러나 맡은 바에 크게 흥미를 느끼지 못했고 제가 정말로 좋아하고 열정적으로 임할 수 있는 일을 찾지 못해 방황했습니다.
- 그러던 중 우연한 기회에 그룹사의 Digital transformation을 이끄는 팀으로 이동하게 되었습니다. 그곳에서 아파트 실거래가 데이터 분석을 통해 자사 신축 아파트의 적정 분양가를 선정하고자 하는 프로젝트를 리드하였고 이후에는 전사의 Tableau 운영과 EIS (중역정보시스템) 구축 등의 업무를 수행했습니다. Tableau에 관해서는 데이터 준비부터 대시보드 설계 및 제작까지 전 과정을 혼자 할 수 있는 정도의 실력을 가지고 있습니다. 그 전까지 저는 코딩, 데이터, 머신 러닝 이런 것들과는 전혀 인연이 없었지만 크게 흥미를 느꼈고 앞으로 이와 관련된 일을 하고 싶다고 확신하게 되었습니다. 컴퓨터 관련 전공자들과 비교했을 때 제 자신이 늦었다고 생각해 그들과의 경쟁에서 지지 않아야겠다는 생각으로 기술적인 역량을 갖추기 위해 끊임없이 노력하고 있습니다.
- 이후 데이터를 좀 더 전문적으로 다루는 기업에서 일하고자 결심하여 현재는 언어 데이터와 번역을 전문으로 하는 'Flitto'라는 기업에서 근무하고 있습니다. 입사 초기에는 고객사에 납품할 데이터셋을 구축하는 데 있어서 머신 러닝 등의 알고리즘을 통해 데이터를 정제하여 수작업을 최소화하고 데이터셋의 품질을 높이는 일을 맡았습니다. 주로 자연어 데이터셋을 구축하였고 음성 데이터셋도 구축해 본 일이 있습니다.
- 올해부터는 같은 회사에서 'Place Lab'이라는 조직으로 이동하여 이미지를 처리하는 알고리즘을 개발하고 있습니다. 회사의 주력 사업 중 하나가 '메뉴 번역'인데 이는 음식점의 메뉴 이미지를 다국어로 번역해 QR 코드로 제공하는 서비스입니다. 원본 이미지에서 번역이 필요한 텍스트를 지우고 기계번역과 전문가 검수 등을 거친 번역문을 렌더링하여 번역된 메뉴 이미지를 만들게 됩니다. 이 중에서 텍스트를 지우는 기술을 개발하여 내부적으로 사용하고 있으며 현재는 텍스트를 렌더링하는 기술을 연구 및 개발하고 있습니다. 텍스트를 지우는 기술은 크게 3가지의 룰 기반 혹은 딥 러닝 기반의 알고리즘이 적용되었으며 비슷한 서비스를 제공하는 타사보다 훨씬 뛰어난 성능을 자랑합니다. 텍스트를 렌더링하는 기술은 원본 이미지로부터 텍스트에 관한 7가지 속성을 인식하고 이를 바탕으로 원본 이미지와 유사한 수준으로 번역문을 렌더링하고자 하는 기술입니다. 기존에 하나의 텍스트 컬러만을 사용하여 작은 폰트 사이즈로 렌더링할 수밖에 없었던 것에 비하면 이미지 번역 품질을 크게 향상시켜 나가고 있습니다.
- 이처럼 저는 데이터 시각화부터 음성, 자연어, 이미지 데이터를 다루는 경험까지 지속적으로 역량을 발전시켜 나가고 있습니다. 지금까지는 주로 작은 규모로 개발을 해왔지만 이제는 큰 규모의 팀을 이루어 더욱 어려운 프로젝트를 수행하며 디지털 기술을 통해서 세상을 풍요롭게 바꾸어 나가고 싶습니다. 특히 메뉴 이미지 처리를 연구하면서 특히 Scene text image에 큰 관심을 갖게 되었습니다. 예를 들어 Text spotting에 있어서 Text detection과 Text recognition의 두 단계로 이루어진 기술로는 성능에 한계가 있으며 인간처럼 자연어와 이미지 모두에 대한 이해를 바탕으로 한 모델로 기술이 발전해 나가야 한다고 생각합니다. Scene text editing에 있어서는 그간의 연구가 주로 동일한 하나의 언어에 대해서 문자 또는 단어 단위로 텍스트를 바꾸는 데에 그치는데, 이 기술의 활용 가능성이 가장 큰 이미지 번역에 있어서는 서로 길이 차이가 큰 텍스트에 대해서도 자연스럽게 편집이 가능한 기술을 연구해보고자 합니다.
## (2) Strengths and Weaknesses
- 자리에서 좀처럼 일어나지 않고 매우 집중하여 일합니다.
- 현실에 안주하지 않고 항상 새로운 것을 배워 실력을 갖추고자 합니다.
- 적을 두지 않고 여러 사람과 두루두루 좋은 관계를 유지하는 편입니다.
- 다른 사람의 의견을 경청하고 원만하게 협의합니다.
