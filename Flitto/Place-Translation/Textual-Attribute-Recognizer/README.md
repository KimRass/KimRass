# 1. Textual Attribute Recognition
- 메뉴 이미지의 각 텍스트에서 7가지 속성을 추출하고 이를 번역문 렌더링 시 활용함으로써 시각적 품질을 향상시킵니다.
    - [1) Font Size](#1-font-size)
    - [2) Writing Direction](#2-writing-direction)
    - [3) Text Alignment](#3-text-alignment)
    - [4) Text Line Breaking](#4-text-line-breaking)
    - [5) Text Color](#5-text-color)
    - [6) Text Border](#6-text-border)
    - [7) Text Border Color](#7-text-border-color)

## 1) Font Size
|Original image|
|:-|
|<img src="https://user-images.githubusercontent.com/67457712/235050769-a7970aa6-2b99-4de0-a20b-1bf301ad7d38.jpg" width="600">|

|Bounding box annotation|
|:-|
|<img src="https://user-images.githubusercontent.com/67457712/235050291-34d76d66-cfdc-47fe-a86a-8ea177eda211.jpg" width="600">|

|Text region score map|
|:-|
|<img src="https://user-images.githubusercontent.com/67457712/235050295-8b42ccdf-976c-4dc3-a0f8-3ceffee8bf7a.jpg" width="600">|
|'CRAFT' scene text detection model [1]을 사용해 'text region score map'을 추출합니다. 어떤 픽셀이 빨간색에 가까울수록 모델이 그 픽셀을 텍스트의 중심이라고 확신함을 나타냅니다.|

|Text region segmentation map|
|:-|
|<img src="https://user-images.githubusercontent.com/67457712/235050300-e4dff000-f476-485f-8cfd-28799b24e9f8.jpg" width="600">|
|<img src="https://user-images.githubusercontent.com/67457712/235050726-ab8d3a2f-75cd-4637-8e02-f2105ba41fff.jpg" width="200">|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/06d1994c-c9a8-495a-a634-27d21b06a2e8" width="600">|
|connected component labeling, watershed 등을 사용해 각 문자를 서로 다른 Class로 구분하는 'text region segmentation map'을 생성합니다. 각 class가 정확히 하나의 문자를 나타내지는 못하므로 pseudo characters라고 부르겠습니다.|
|매우 많은 수의 label이 생성되지만 이해를 돕기 위해 26개의 색상을 사용하여 단순화했습니다.|
|각 pseudo character의 크기에 일정한 값을 곱해 이를 통해 해당 문자의 font size로 구합니다. 실제로는 하나의 bounding box에서 다양한 font size가 존재할 수도 있지만 모두 동일하다고 가정하여 통계적인 방법을 통해 하나의 font size를 추출합니다.|

## 2) Writing Direction
|PCCs|
|:-|
|<img src="https://user-images.githubusercontent.com/67457712/235050297-a43a5b3c-fdb1-41ad-a30c-30e0f2778910.jpg" width="600">|
|text region segmentation map으로부터 pseudo character centers (PCCs) [3]를 추출합니다.|
|각 PCC에 대해서 가장 가까운 다른 PCC를 찾습니다. 두 PCCs간의 x축 방향과 y축 방향 각각에 대한 거리를 구해서 x축 방향의 거리가 y축 방향의 거리보다 더 멀면 가로쓰기라고 판단하고 y축 방향의 거리가 x축 방향의 거리보다 멀면 세로쓰기라고 판단합니다.|
|이때 가장 가까운 다른 PCC와의 거리가 font size와 비교해 너무 작다면, 바로 그 다음으로 가까운 PCC를 찾습니다. 이는 동일한 하나의 문자에 대해서 다수의 PCCs가 추출되는 경우에 이를 보완하기 위함입니다.|

## 3) Text Alignment
|Ground truth|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/1d4e8384-926c-4de0-bde4-c8517312bdf3" width="500">|
|rule-based approach로는 추출이 불가능하다고 판단했습니다 [2]. 또한 각 bounding box의 이미지 영역만을 보고는 알 수 없으며 이미지의 전체적인 visual features를 고려하여야만 높은 정확도로 판단할 수 있다고 보았습니다.|
|이에 4개의 class ('none', 'left', 'center', 'right')에 대한 semantic segmentation 문제로 접근했습니다. 가로쓰기 텍스트에 대해서는 'left', 'center', 'right' 중 하나로, 세로쓰기 텍스트에 대해서는 'none'으로 레이블링을 했습니다. 즉 가로쓰기 텍스트에 한해서만 모델이 text alignment를 예측하도록 학습시켰습니다.|
|빨간색: 'left', 초록색: 'center', 파란색: 'right', 그 외: 'none'|
|모델이 내놓은 결과에 대해서, 각 bounding box가 차지하는 영역에서 가장 많은 픽셀 수를 차지하는 class를 통해 text alignment를 예측합니다.|
<!-- - 1,000장의 이미지, 19,523개의 bounding box에 대해 학습한 결과 mIoU 0.7341 -->

|Example 1|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/eab61871-b37f-487b-b12b-9059aefcabf6" width="800">|

|Example 2|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/d27aba57-f3ba-405e-a3f2-157b6e0fa55d" width="800">|
<!-- - <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/67457712/287581756-d2460fbf-9716-4b52-b2c6-0695264a4d33.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240404T000439Z&X-Amz-Expires=300&X-Amz-Signature=9f3bcc9523004783714da96d1b10865e8631f7aba8baad1620dab455a74f92ce&X-Amz-SignedHeaders=host&actor_id=67457712&key_id=0&repo_id=669664726" width="800"> -->

## 4) Text Line Breaking
- 주어진 텍스트를 렌더링할 때 어디에 줄바꿈을 삽입했을 때 font size가 가장 원본에 가까워질 지를 계산합니다.
<!-- - 주어진 텍스트를 렌더링할 때 한 줄로 할 수도 있고 중간에 줄바꿈을 삽입하여 두 줄 이상으로 할 수도 있을 것입니다. 수많은 경우 중 최적을 찾는 알고리즘입니다. -->

### (1) 띄어쓰기가 있는 언어 (ko, en, vi, id, ms, es)
- 어절과 어절 사이에만 줄바꿈을 삽입할지 여부가 결정됩니다.
- 한국어 예시
   |Case|Number of lines|Max. # of<br>characters<br>in a single line|
   |------|---|---|
   |저희 업소에서는 남은 음식물을 재활용하지 않습니다.|1|28|
   |저희 업소에서는 남은 음식물을 재활용하지<br>않습니다.|2|22|
   |저희 업소에서는 남은 음식물을<br>재활용하지 않습니다.|2|16|
   |저희 업소에서는 남은<br>음식물을 재활용하지<br>않습니다.|3|11|
   |저희 업소에서는<br>남은 음식물을<br>재활용하지<br>않습니다.|4|8|
   |저희<br>업소에서는<br>남은 음식물을<br>재활용하지<br>않습니다.|5|7|

### (2) 띄어쓰기가 없는 언어 (ja, zh-cn, zh-tw, th)
- part-of-speech tagging에 기반하여 의미 단위로 텍스트를 분리하고 줄의 시작 또는 끝에 올 수 없는 문자를 고려하여 줄바꿈합니다.
- 일본어 예시
   |Case|Number of lines|Max. # of<br>characters<br>in a single line|
   |------|---|---|
   |当店では残飯を再活用しません。|1|15|
   |当店では残飯を再活用<br>しません。|2|10|
   |当店では残飯を再<br>活用しません。|2|8|
   |当店では残飯を<br>再活用<br>しません。|3|7|
   |当店では<br>残飯を再活用<br>しません。|3|6|
   |当店では<br>残飯を再<br>活用<br>しません。|4|5|
- 중국어 예시
   |Case|Number of lines|Max. # of<br>characters<br>in a single line|
   |------|---|---|
   |本店绝不使用剩菜。|1|9|
   |本店绝不使用剩<br>菜。|2|7|
   |本店绝不使用<br>剩菜。|2|6|
   |本店绝不<br>使用剩菜。|2|5|
   |本店绝<br>不使用<br>剩菜。|3|3|
- 위 경우 각각에 대해, 추출된 font size로 bounding box를 벗어나지 않도록 렌더링을 시도하고 bounding box를 벗어난다면 벗어나지 않도록 font size를 줄입니다.
- 가장 font size가 클 수 있는 줄바꿈을 최종적으로 선택하고 같은 font size에 대해서는 줄바꿈 횟수가 가장 적은 것을 선택합니다. 

## 5) Text Color
|Scene text removal|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/dbc56f19-2e3e-4280-884d-849c1ff611d1" width="600">|

|Per-pixel absolute difference|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/8a9c2b08-d3df-4304-979f-78435c33055e" width="600">|
|원본 이미지와 텍스트가 제거된 이미지 사이의 픽셀 단위의 차이를 구합니다.|

|Mask|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/0c8565c8-dfb2-4170-8720-02c00a78acaa" width="600">|
|이를 바탕으로 mask를 생성합니다 [2].|

- Color extraction
    - 각 bounding box에 대해서 mask에 해당하는 영역에 대해서만 원본 이미지로부터 픽셀별 색깔을 조사합니다. 이때 비슷하지만 서로 조금씩 다른 색깔은 많은 픽셀 수를 갖는 색깔 쪽으로 통합해 나갑니다.
    - 색깔의 수가 3개가 될 때까지 통합합니다. 이 3개의 색깔 중에서 가장 많은 픽셀 수를 갖는 `(7, 145, 58)`을 text color로 정합니다.

|Original image|Per-pixel absolute difference|Text color candidates|
|:-|:-|:-|
|<img src="https://user-images.githubusercontent.com/67457712/235061924-e372749d-fa8f-4db2-a655-ad5210ff0c88.jpg" width="400">|<img src="https://user-images.githubusercontent.com/67457712/235061927-f1e4e5ff-fcc4-4640-9817-4d14ba467e28.jpg" width="400">|<img src="https://user-images.githubusercontent.com/67457712/235064154-1ff39941-f237-4639-8735-db7bf116986a.jpg" width="300">|

## 6) Text Border
|Text border|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/7dbc77da-cdeb-48d0-bc0e-78ea0bd9024b" width="200">|
|text border를 적용하지 않았을 때, 사람이 그 텍스트를 읽을 수 없으면 text border를 적용하고 읽을 수 있다면 적용하지 않습니다.|
|font size, text alignment, text line breaking이 결정되면 각 문자가 어디에 렌더링될 지가 결정됩니다. 각 문자에 대해서, 추출된 text color와 text stroke를 둘러싼 영역의 픽셀 하나하나 사이의 contrast ratio를 계산합니다. 이 값이 일정한 값 미만이라면 그 두 색깔 조합은 가독성이 좋지 않은 것으로 판단합니다.|

|Text readibility|
|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/672349f4-3781-4031-ba7c-046a7779c189" width="900">|
|각 문자에 대해서, text stroke를 둘러싼 영역의 전체 픽셀 수와 비교하여 가독성이 좋지 않은 픽셀의 수를 계산합니다. 이 값은 그 문자의 영역 중 읽을 수 있는 부분의 비율을 나타낸다고 볼 수 있습니다.|
|이 값을 그 문자의 readability라고 할 때, readability가 어떤 값 미만인 문자가 1개라도 존재할 경우 (1개의 문자라도 가독성이 좋지 않은 것이 있다면) 그 텍스트에는 text border를 적용합니다.|

## 7) Text Border Color
- 검은색과 하얀색 중, 추출된 text color의 보색과 contrast ratio가 더 큰 쪽을 선택합니다. 이것을 A라 합니다.
- CIELAB 색 공간 상에서 A와, 추출된 text color의 보색과의 linear interpolation을 수행한 값을 text border color로 지정합니다. 이렇게 하는 이유는 가독성을 높이기 위해 text border의 보색을 text border로서 사용함과 동시에 text border color가 너무 돋보이지 않도록 하기 위함입니다.

# 2. Examples

## 1) '화정 by 카쿠시타'
|Original image|Textual attribute recogition (en, ja and zh-cn)|
|:-|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/b7d3c121-a5e9-4f07-a049-3de63cc18a50" width="250">|<img src="https://github.com/KimRass/KimRass/assets/67457712/927c8445-4ce8-4690-84cf-664437fdf48b" width="750">|

## 2) '어반 웍 스테이션'
|Original image|Textual attribute recogition (en, ja and zh-cn)|
|:-|:-|
|<img src="https://github.com/KimRass/KimRass/assets/67457712/e5705985-471d-46d0-8a1c-ed69c584ae01" width="200">|<img src="https://github.com/KimRass/KimRass/assets/67457712/7dfb4c98-d988-43b7-a36c-23fc8a325b83" width="600">|

# 3. Improvements
<table>
    <thead>
        <tr>
            <th>Textual attribute</th>
            <th>Before</th>
            <th>After</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">Font size</td>
            <td>bounding box의 가로와 세로 길이에 의해 결정되므로 원본 font size와 크게 달라질 수 있음.</td>
            <td>(충분한 텍스트 렌더링 공간이 확보된다면) bounding box의 가로와 세로 길이에 무관.</td>
        </tr>
        <tr>
            <td>원본 font size보다 커질 수 있음.</td>
            <td>원본 font size를 넘지 않는 한도 내에서 주어진 bounding box를 최대한 활용하는 font size 추출.</td>
        </tr>
        <tr>
            <td>Writing direction</td>
            <td>bounding box의 가로와 세로 길이의 비율에 의해 결정되므로 정확성이 떨어짐.</td>
            <td>정확성이 높음.</td>
        </tr>
        <tr>
            <td>Text alignment</td>
            <td>모든 가로쓰기 텍스트에 left alignment 적용.</td>
            <td>가로쓰기 텍스트에 대해 적절히 정렬하여 시각적 품질 향상.</td>
        </tr>
        <tr>
            <td>Text line breaking</td>
            <td>띄어쓰기가 없는 언어의 경우, 텍스트의 의미와 무관하게 줄바꿈이 삽입되는 경우가 많음.</td>
            <td>텍스트의 의미를 고려하여 적절한 위치에 줄바꿈 삽입.</td>
        </tr>
        <tr>
            <td>Text color</td>
            <td>검은색 또는 하얀색으로 단조로움.</td>
            <td>원본 이미지의 text color를 반영하여 다채롭고 생동감 있는 느낌을 전달.</td>
        </tr>
        <tr>
            <td rowspan="2">Text border</td>
            <td>모든 텍스트에 text border가 적용되어 가독성은 보장되나 심미성이 떨어짐.</td>
            <td>가독성을 위해 필요한 경우에만 적용하여 심미성 향상.</td>
        </tr>
        <tr>
            <td>CSS의 text-stroke 속성을 통해 구현되어 글자가 빛나는 듯한 효과.</td>
            <td>CSS의 text-shadow 속성을 통해 구현되어 깔끔한 느낌을 전달.</td>
        </tr>
        <tr>
            <td>Text border color</td>
            <td>검은색 텍스트일 경우 하얀색, 반대의 경우 검은색 사용으로 text border가 지나치게 강조됨.</td>
            <td>추출된 text color와의 interpolation을 통해 강조가 완화됨.</td>
        </tr>
    </tbody>
</table>

# 4. References
- [1] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [2] ['더' 잘 읽히고 자연스러운 이미지 번역을 위해 (파파고 텍스트 렌더링 개발기)](https://deview.kr/2023/sessions)
- [3] [CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://github.com/KimRass/KimRass/tree/main/Flitto/papers/cleval_character_level_evaluation_for_text_detection_and_recognition_tasks.pdf)

<!-- ## Font Size Clustering and Limitation
- Font size와 Text color (R, G, B), 4개의 Features를 가지고 K-means clustering
    - Number of clusters: 2, 3, 4, 5
        - <img src="https://github.com/KimRass/scene_text_renderer/assets/105417680/d20a4653-0783-4967-bb67-5ff1b00528b7" width="1200">
2. Cluster 내에서 Font size의 평균을 구하고, 평균보다 큰 Font size는 평균으로 감소시키고 평균보다 작은 Font size는 그대로 둡니다.
    - Before and after performing font size clustering and limitation
        - <img src="https://github.com/KimRass/transformer_based_models/assets/105417680/98f557d2-1391-4249-93f3-de8a2a45911e" width="900">
        - <img src="https://github.com/KimRass/scene_text_renderer/assets/105417680/649e47d3-7a8d-44cd-97ae-b5904850a34a" width="900">
        - <img src="https://github.com/KimRass/scene_text_renderer/assets/105417680/538be647-eb09-4f81-8c88-244505045066" width="700"> -->
