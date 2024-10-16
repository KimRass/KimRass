# 1. Overview
- 'N' 사의 OCR 모델 구축 및 평가를 위한 아랍어 텍스트가 포함된 scene text images를 플리토의 크라우드 데이터 레이블링 플랫폼을 통해 수집합니다.
- 높은 품질과 다양성을 갖춘 데이터를 수집하기 위해서는 수동 QC가 필요하지만 데이터 수집에 드는 비용을 절감하기 위해서 그 중 일부를 자동화하는 알고리즘을 개발했습니다. 자동 QC를 통과한 데이터에 한해서만 수동 QC를 진행했습니다.

# 2. Data Requirements
1. 이미지의 가로와 세로 중 긴 쪽이 1,280px 이상일 것. (이미지의 해상도가 너무 작을 경우 scene text가 식별하기 어려울 정도로 작을 가능성이 높음.)
2. ***이미지의 EXIF를 삭제하고 이미지를 올바른 방향으로 회전시킬 것.***
    - ***이를 만족하기 위해서 이미지의 회전 방향을 분류하는 모델 ('Image Rotation Classifier')을 사용합니다.***
3. ***text bounding boxes의 수가 일정 범위 내이고 아랍어의 비율이 80% 이상일 것.***
    - ***'EasyOCR' OCR framework를 사용하여 OCR을 수행합니다.***
    - ***아랍어는 띄어쓰기가 있으므로 공백 (whitespace)으로 구분된 어절을 단위로 text bounding box annotation을 수행해야 합니다. 즉 하나의 text bounding box에 포함된 텍스트에는 공백이 존재할 수 없습니다.***
    - ***text bounding box별로 텍스트가 어떤 언어인지 알 수 있다면 이를 가지고 아랍어의 비율을 구할 수 있습니다. 이를 위해서 텍스트가 어떤 언어로 되어있는지 분류하는 모델 ('Language Identifier')을 사용합니다.***
4. ***배경 또는 텍스트의 레이아웃이 서로 유사한 이미지가 없을 것. (데이터의 다양성을 확보하여 이를 바탕으로 구축된 모델의 일반화 능력을 보장하기 위함.)***
    - ***이를 만족하기 위해서 image embedding을 기반으로 이미지간 유사도를 판별할 수 있는 모델을 사용합니다.***
5. 단순한 배경에 텍스트만 존재하는 이미지 또는 디스플레이를 촬영하거나 캡쳐한 이미지가 아닐 것.
6. 텍스트가 사람이 읽을 수 있을 것.
7. 개인정보나 저작권법상 보호받는 저작물을 포함하지 않을 것.
<!-- 8. 아랍어의 모음을 포함.
    - 아랍어에는 우리말의 초성체처럼 모음을 표기하지 않거나 표기하는 두 가지 표기법이 존재합니다.
    - 아랍어에서는 우리말의 초성체처럼 모음을 표기하지 않는 경우가 많아 데이터의 다양성을 위해서 모음을 표기하는 텍스트가 일정 비율 이상이 되는 것이 요구조건 중 하나였습니다. 이미지에서 하나의 텍스트에서라도 모음이 있다면 모든 텍스트가 모음이 포함된 표기를 할 것이라고 가정했습니다.
    - 이를 만족하기 위해서 '모음이 포함된 이미지'라는 별도의 카테고리를 만들었습니다. -->

# 3. Process
1. 유저가 전용 앱을 통해 이미지의 카테로기를 선택하고 scene text image를 업로드합니다.
2. 'Google vision OCR'을 통해 부적절한 이미지를 거릅니다.
3. 이미지의 해상도가 적절한지 확인합니다. (Data requirements #1)
4. 이미지의 EXIF를 제거하고 Image Rotaion Classifier를 사용해 이미지를 올바른 방향으로 회전시킵니다. (Data requirements #2)
5. EasyOCR을 사용해서 OCR을 수행합니다.
6. Language Identifier를 사용해서 각 text bounding box의 텍스트가 어떤 언어인지 확인합니다. (Data requirements #3)
7. 역대 수집된 모든 이미지들 중에서 유사한 이미지가 없는지 확인합니다. (Data requirements #4)

# 4. Future Improvements
- 텍스트가 아닌 scene text로부터 바로 언어를 분류합니다.


<!-- # 3. Image Categories
- 모음이 포함된 이미지
- 가게 간판
- 표지판 (교통 표지판, 안내문 등)
- 메뉴 (인쇄물, 손글씨 등)
- 잡지 커버
- 광고 (포스터, 전단지 등)
- 제품 포장(뒷면) : 영양/제품 정보 등
- 제품 포장(앞면) : 병, 제품 상자 등 -->

<!-- 
# Process
1. 이미지 접수
2. Google vision OCR
    - 부적절한 이미지 반려
3. 자동 QC
    1. Fail 없음
        1. 모음 포함 카테고리에 속하면 모음 포함 시 자동 QC 통과
        2. 모음 포함 카테고리에 속하지 않으면 자동 QC 통과
    2. Fail 존재
        1. 자동 QC 기각
        2. (2) 박스 과다, (7) OCR confidence 낮음: 수동 QC 대기 상태 돌입
        3. 나머지: 종료

1. 이미지 접수 (`qc_pipeline`)
    - image_url	string	null	분석할 이미지의 s3 URL
    - image_id	int	null	이미지의 고유 id
    - image_src	string	null	이미지 수집 소스: "arcade", "b2b", "vender" 등
    - project	string	null	이미지 수집 프로젝트명 "naver_arabic_image" 등
    - target_lang	string	"ar"	분석 대상 언어
    - similarity	float	0.95	이미지 유사도 점수 제한치 (0.0에서 1.0까지)
        - 같은 project 내의 유사 이미지로 검색 제한
    - lang_ratio	float	0.5	대상 언어 박스 비율 (0.0에서 1.0까지)
    - rotation	boolean / int	true	QC 중 이미지 회전 적용 유무
        - true: 방위 탐지 후 회전 적용
        - false  : 회전 미적용
        - 정수값(0, 90, 180, 270): 해당 방위로 이미지 회전(시계방향) 후 QC
        - (0 = false )
    - n_box_range	array of integers	[1, 10000]	상자의 최소 및 최대 수
    - box_info	boolean	false	상자 별 세부 정보 요청
1. QC (`perform_image_qc`)
    - Remove image metadata
    - Perform rotation recognition and OCR
    - `if rotation in [False, 0, 90, 180, 270]`:
        - 해당 각도 (`False`일 경우 0도)로 이미지 회전 후 OCR 구동
    - 그 외
        - rotation recognition 후 4 방향 OCR
        ```python
        pred_angle, rot_score = self.check_rotation(image)
        test_angles = [
            pred_angle,
            (pred_angle + 180) % 360,
            (pred_angle + 90) % 360,
            (pred_angle + 270) % 360,
        ]
        ```
        - 4 방향 OCR 중 score가 0.5 이상인 각도가 나오면 중단하고 이 각도를 `best_angle`로 지정
1. Update
    - `best_angle`에 따라 회전된 이미지를 DB에 삽입 또는 갱신 & 이 이미지의 임베딩이 추가된 fai
    - 기존 DB에 저장된 이미지와 유사도 검사
1. Back-up -->

<!-- 
# 설명 듣기
- 애초에 네이버 파파고의 아랍어 이미지 수집 프로젝트를 염두에 두고 시작했던 프로젝트
- 납품 물량 한 달에 10,000장
- 요구조건
    - 다양한 배경!!
    - 장축 1,280px
    - OCR (박싱 + 전사)
    - 박스 20 ~ 30개 (단어별로)
    - 텍스트는 언어 관계 없이 전부 다 디텍션
    - 아랍어 외는 돈 케어로 전사
    - 개인정보, 저작권
- 모국어가 아랍어인 유저만 참가 가능

- 텍스트
    - 대화하기
    - 전사
    - 번역
- 음성
    - 이벤트

- 이미지 업로드하면 바로 자동 QC 보냄
- 이때 안 돌아갔으면 매일 아침 정해진 시간에 자동 QC 보냄 -->
