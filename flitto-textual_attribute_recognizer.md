# 1. 문제 정의
- 기존 이미지 번역 결과물에 있어서, 번역 자체의 품질과 무관하게 이미지의 시각적 품질이 'N'사의 서비스 대비 매우 떨어진다고 판단했습니다. 문제점을 정리하면 다음과 같습니다.
    - 너무 작거나 큰 폰트 사이즈:
        - 단순히 바운딩 박스 크기에 따라 폰트 사이즈를 계산하는 알고리즘을 사용했습니다.
        - 이로 인해 품질이 바운딩 박스 생성 작업자의 역량에 크게 좌우된다는 문제점이 있었습니다.
    - 잘못된 텍스트 방향:
        - 텍스트 방향은 가로쓰기 또는 세로쓰기를 말합니다.
        - 바운딩 박스의 가로 길이와 세로 길이를 서로 비교해 텍스트 방향을 계산하는 아주 단순한 알고리즘을 사용했습니다.
        - 대부분의 경우에 알고리즘이 잘 작동했지만 그렇지 못했을 경우 사용자 경험을 크게 저해했으며 품질이 바운딩 박스 생성 작업자의 역량에 크게 좌우된다는 점이 문제였습니다.
    - 모두 왼쪽으로만 정렬된 텍스트:
        - 대부분의 경우 가로쓰기 텍스트는 왼쪽 정렬, 세로쓰기 텍스트는 위쪽 정렬이지만, 일부 이미지에서 모든 텍스트에 대해서 왼쪽 정렬 또는 위쪽 정렬만을 사용하면 원본 이미지의 느낌에서 크게 벗어났습니다.
    - 아무 곳에나 삽입된 줄바꿈:
        - 특히 띄어쓰기가 없는 언어 (일본어, 중국어 등)의 경우 하나의 의미 덩어리가 두 줄에 걸쳐 렌더링되는 경우가 많았습니다. 이는 각 언어의 사용 규칙에도 벗어나므로 사용자 경험을 저해했습니다.
    - 단조로운 텍스트 및 테두리 색상:
        - 어떤 배경에서도 텍스트의 가독성을 보장하기 위해서 텍스트 및 테두리의 색상을 각각 검은색과 흰색 또는 흰색과 검은색으로 했습니다.
        - 원본 이미지에서 다양한 색상이 사용된 경우, 이미지 번역 결과물의 단조로움은 더욱 대비됐습니다.
    - 모든 텍스트에 삽입된 테두리:
        - 마찬가지로 어떤 배경에서도 텍스트의 가독성을 보장하기 위해 모든 텍스트에 테두리를 설정했습니다.
        - 원본 이미지에서 대부분의 텍스트에는 테두리가 없으므로, 결과물이 원본 이미지의 느낌에서 크게 벗어나는 문제점이 있었습니다.
- 이에 원본 이미지의 텍스트에서 그 속성을 추출하는 모델을 개발했습니다.

# 2. 문제 해결
|원본 이미지|바운딩 박스|
|-|-|
|<img src="https://user-images.githubusercontent.com/67457712/235050769-a7970aa6-2b99-4de0-a20b-1bf301ad7d38.jpg" width="300">|<img src="https://user-images.githubusercontent.com/67457712/235050291-34d76d66-cfdc-47fe-a86a-8ea177eda211.jpg" width="300">|

## 1) 폰트 크기
- 'CRAFT' [1]의 출력을 시각화한 이미지입니다. 빨간색에 가까울수록 그 픽셀이 문자의 중심에 가까움을 의미합니다. 이를 'text region score map'이라고 부르겠습니다.
    |Text Region Score Map|
    |-|
    |<img src="https://user-images.githubusercontent.com/67457712/235050295-8b42ccdf-976c-4dc3-a0f8-3ceffee8bf7a.jpg" width="600">|
- 'text stroke mask'에 connected component labeling (CCL), watershed 등을 활용하여 서로 다른 문자를 구분하는 'text region segmentation map'을 생성합니다.
    |Text region segmentation map|
    |-|
    |<img src="https://user-images.githubusercontent.com/67457712/235050300-e4dff000-f476-485f-8cfd-28799b24e9f8.jpg" width="600">|
    |<img src="https://user-images.githubusercontent.com/67457712/235050726-ab8d3a2f-75cd-4637-8e02-f2105ba41fff.jpg" width="200">|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/06d1994c-c9a8-495a-a634-27d21b06a2e8" width="600">|
- 'text region segmentation map'의 각 레이블의 높이를 구하고 이로부터 통계적인 방법을 통해 바운딩 박스별로 하나의 폰트 사이즈를 추출합니다.
- 실제로는 하나의 바운딩 박스 안에서 다양한 폰트 사이즈가 존재할 수도 있지만 모두 동일하다고 가정합니다.

## 2) 텍스트 방향
- 'text region score map'을 사용해 각 문자의 중심 (Pseudo Character Center (PCC) [3])을 추출합니다.
    |Pseudo Character Centers (PCCs)|
    |-|
    |<img src="https://user-images.githubusercontent.com/67457712/235050297-a43a5b3c-fdb1-41ad-a30c-30e0f2778910.jpg" width="600">|
- 각 PCC에 대해서 가장 가까운 다른 PCC를 찾습니다. 두 PCC간의 x축 방향과 y축 방향 각각에 대한 거리를 구해서 x축 방향의 거리가 y축 방향의 거리보다 더 멀면 가로쓰기라고 판단하고 y축 방향의 거리가 x축 방향의 거리보다 멀면 세로쓰기라고 판단합니다.
- 이때 가장 가까운 다른 PCC와의 거리가 폰트 사이즈와 비교해 너무 작다면, 바로 그 다음으로 가까운 PCC를 찾습니다. 이는 동일한 하나의 문자에 대해서 다수의 PCC가 추출되는 에러를 보완하기 위함입니다.

## 3) 텍스트 정렬
- 규칙 기반 방법론으로는 추출이 불가능하다고 판단했습니다 [2]. 또한 각 텍스트만 보고는 알 수 없고 이미지 전체를 봐야만 높은 정확도로 판단할 수 있다고 보았습니다.
- 이에 4개의 클래스 ('none', 'left', 'center', 'right')에 대한 semantic segmentation 문제로 접근했습니다. 세로쓰기 텍스트는 데이터의 수가 매우 부족했기 때문에 전부 위쪽 정렬을 적용하고 가로쓰기 텍스트에 한해서만 모델을 통해 텍스트 정렬을 추출하도록 했습니다.
    |정답 데이터|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/1d4e8384-926c-4de0-bde4-c8517312bdf3" width="600">|
    |빨간색: 'left', 초록색: 'center', 파란색: 'right'|
- 모델의 출력에서 바운딩 박스별로 가장 많은 픽셀 수를 차지하는 클래스를 텍스트 정렬로 추출합니다.
    |원본|As-is|To-be|
    |-|-|-|
    |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/125_257-ori.png" width="600">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/125_257-as_is.png" width="600">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/125_257-to_be.png" width="600">|
    <!-- |<img src="https://github.com/KimRass/KimRass/assets/67457712/d27aba57-f3ba-405e-a3f2-157b6e0fa55d" width="600">| -->

## 4) 텍스트 줄바꿈
- 주어진 텍스트를 렌더링할 때 어디에 줄바꿈을 삽입했을 때 폰트 사이즈가 가장 원본에 가까워질 지를 계산합니다.
<!-- - 주어진 텍스트를 렌더링할 때 한 줄로 할 수도 있고 중간에 줄바꿈을 삽입하여 두 줄 이상으로 할 수도 있을 것입니다. 수많은 경우 중 최적을 찾는 알고리즘입니다. -->

### (1) 띄어쓰기가 있는 언어 (한국어, 영어, 베트남어, 인도네시아어, 말레이어, 스페인어)
- 어절과 어절 사이에만 줄바꿈을 삽입할지 여부가 결정됩니다.
- 한국어 예시:
   |<div style="width:300px">구분</div>|줄 수|한 줄당<br>문자의<br>최대 개수|
   |------|---|---|
   |저희 업소에서는 남은 음식물을 재활용하지 않습니다.|1|28|
   |저희 업소에서는 남은 음식물을 재활용하지<span style="color:red">\n</span>않습니다.|2|22|
   |저희 업소에서는 남은 음식물을<span style="color:red">\n</span>재활용하지 않습니다.|2|16|
   |저희 업소에서는 남은<span style="color:red">\n</span>음식물을 재활용하지<span style="color:red">\n</span>않습니다.|3|11|
   |저희 업소에서는<span style="color:red">\n</span>남은 음식물을<span style="color:red">\n</span>재활용하지<span style="color:red">\n</span>않습니다.|4|8|
   |저희<span style="color:red">\n</span>업소에서는<span style="color:red">\n</span>남은 음식물을<span style="color:red">\n</span>재활용하지<span style="color:red">\n</span>않습니다.|5|7|

### (2) 띄어쓰기가 없는 언어 (일본어, 중국어 간체, 중국어 번체, 태국어)
- 품사 분석에 기반하여 의미 단위로 텍스트를 분리하고 줄의 시작 또는 끝에 올 수 없는 문자를 고려하여 줄바꿈합니다.
- 일본어 예시:
   |<div style="width:300px">구분</div>|줄 수|한 줄당<br>문자의<br>최대 개수|
   |------|---|---|
   |当店では残飯を再活用しません。|1|15|
   |当店では残飯を再活用<span style="color:red">\n</span>しません。|2|10|
   |当店では残飯を再<span style="color:red">\n</span>活用しません。|2|8|
   |当店では残飯を<span style="color:red">\n</span>再活用<span style="color:red">\n</span>しません。|3|7|
   |当店では<span style="color:red">\n</span>残飯を再活用<span style="color:red">\n</span>しません。|3|6|
   |当店では<span style="color:red">\n</span>残飯を再<span style="color:red">\n</span>活用<span style="color:red">\n</span>しません。|4|5|
- 중국어 예시:
   |<div style="width:300px">구분</div>|줄 수|한 줄당<br>문자의<br>최대 개수|
   |------|---|---|
   |本店绝不使用剩菜。|1|9|
   |本店绝不使用剩<span style="color:red">\n</span>菜。|2|7|
   |本店绝不使用<span style="color:red">\n</span>剩菜。|2|6|
   |本店绝不<span style="color:red">\n</span>使用剩菜。|2|5|
   |本店绝<span style="color:red">\n</span>不使用<span style="color:red">\n</span>剩菜。|3|3|
- 위 경우 각각에 대해, 추출된 폰트 사이즈로 바운딩 박스를 벗어나지 않도록 렌더링을 시도하고 바운딩 박스를 벗어난다면 벗어나지 않도록 폰트 사이즈를 줄입니다.
- 가장 폰트 사이즈가 클 수 있는 줄바꿈을 최종적으로 선택하고 같은 폰트 사이즈에 대해서는 줄바꿈 횟수가 가장 적은 것을 선택합니다. 

## 5) 텍스트 색상
- 원본 이미지와 텍스트가 제거된 이미지 사이의 픽셀별 차를 구하고 이를 바탕으로 마스크를 생성합니다 [2].
    |마스크|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/0c8565c8-dfb2-4170-8720-02c00a78acaa" width="600">|
- 각 바운딩 박스에 내의 마스킹 영역에서 원본 이미지로부터 픽셀별 색상을 분석합니다. 이때 비슷하지만 서로 조금씩 다른 색상은 더 많은 픽셀 수를 갖는 색상 쪽으로 통합해 나갑니다. 3가지 색상만 남도록 통합한 다음, 이 중에서 가장 많은 픽셀 수를 갖는 색상을 텍스트 색상으로 정합니다.
    |원본 이미지|픽셀별 차|최종 통합된 3가지 색상|
    |-|-|-|
    |<img src="https://user-images.githubusercontent.com/67457712/235061924-e372749d-fa8f-4db2-a655-ad5210ff0c88.jpg" width="200">|<img src="https://user-images.githubusercontent.com/67457712/235061927-f1e4e5ff-fcc4-4640-9817-4d14ba467e28.jpg" width="200">|<img src="https://user-images.githubusercontent.com/67457712/235064154-1ff39941-f237-4639-8735-db7bf116986a.jpg" width="200">|

## 6) 텍스트 테두리 색상
- 검은색과 하얀색 중, 추출된 텍스트 색상의 보색과 contrast ratio가 더 큰 쪽을 선택합니다. 이것을 A라 합니다.
- CIELAB 색 공간 상에서 A와, 추출된 텍스트 색상의 보색과의 linear interpolation을 수행한 값을 텍스트 테두리 색상으로 지정합니다. 이렇게 하는 이유는 가독성을 높이기 위해 텍스트 테두리의 보색을 텍스트 테두리로서 사용함과 동시에 텍스트 테두리 색상가 너무 돋보이지 않도록 하기 위함입니다.

## 7) 텍스트 테두리 두께
- 텍스트 테두리를 적용하지 않았을 때, 사람이 그 텍스트를 읽을 수 없으면 텍스트 테두리를 적용하고 읽을 수 있다면 적용하지 않습니다.
- 폰트 사이즈, text alignment, text line breaking이 결정되면 각 문자가 어디에 렌더링될 지가 결정됩니다. 각 문자에 대해서, 추출된 텍스트 색상와 text stroke를 둘러싼 영역의 픽셀 하나하나 사이의 contrast ratio를 계산합니다. 이 값이 일정한 값 미만이라면 그 두 색상 조합은 가독성이 좋지 않은 것으로 판단합니다.
    |Text border|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/7dbc77da-cdeb-48d0-bc0e-78ea0bd9024b" width="200">|
- 각 문자에 대해서, text stroke를 둘러싼 영역의 전체 픽셀 수와 비교하여 가독성이 좋지 않은 픽셀의 수를 계산합니다. 이 값은 그 문자의 영역 중 읽을 수 있는 부분의 비율을 나타낸다고 볼 수 있습니다.
- 이 값을 그 문자의 readability라고 할 때, readability가 어떤 값 미만인 문자가 1개라도 존재할 경우 (1개의 문자라도 가독성이 좋지 않은 것이 있다면) 그 텍스트에는 텍스트 테두리를 적용합니다.
    |Text readibility|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/672349f4-3781-4031-ba7c-046a7779c189" width="900">|

# 3. 성과

## 1) 예시
원본|영어|일본어|중국어|
|-|-|-|-|
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_ko_original.jpg" width="250">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_en_after.png" width="250">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_ja_after.png" width="250">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_zh-cn_after.png" width="250">|
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_ko_original.jpg" width="250">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_en_after.png" width="250">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_ja_after.png" width="250">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_zh-cn_after.png" width="250">|

## 2) 개선 사항
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
            <td>바운딩 박스의 가로와 세로 길이에 의해 결정되므로 원본 폰트 사이즈와 크게 달라질 수 있음.</td>
            <td>(충분한 텍스트 렌더링 공간이 확보된다면) 바운딩 박스의 가로와 세로 길이에 무관.</td>
        </tr>
        <tr>
            <td>원본 폰트 사이즈보다 커질 수 있음.</td>
            <td>원본 폰트 사이즈를 넘지 않는 한도 내에서 주어진 바운딩 박스를 최대한 활용하는 폰트 사이즈 추출.</td>
        </tr>
        <tr>
            <td>Writing direction</td>
            <td>바운딩 박스의 가로와 세로 길이의 비율에 의해 결정되므로 정확성이 떨어짐.</td>
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
            <td>원본 이미지의 텍스트 색상를 반영하여 다채롭고 생동감 있는 느낌을 전달.</td>
        </tr>
        <tr>
            <td rowspan="2">Text border</td>
            <td>모든 텍스트에 텍스트 테두리가 적용되어 가독성은 보장되나 심미성이 떨어짐.</td>
            <td>가독성을 위해 필요한 경우에만 적용하여 심미성 향상.</td>
        </tr>
        <tr>
            <td>CSS의 text-stroke 속성을 통해 구현되어 글자가 빛나는 듯한 효과.</td>
            <td>CSS의 text-shadow 속성을 통해 구현되어 깔끔한 느낌을 전달.</td>
        </tr>
        <tr>
            <td>Text border color</td>
            <td>검은색 텍스트일 경우 하얀색, 반대의 경우 검은색 사용으로 텍스트 테두리가 지나치게 강조됨.</td>
            <td>추출된 텍스트 색상와의 interpolation을 통해 강조가 완화됨.</td>
        </tr>
    </tbody>
</table>

# 4. References
- [1] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [2] ['더' 잘 읽히고 자연스러운 이미지 번역을 위해 (파파고 텍스트 렌더링 개발기)](https://deview.kr/2023/sessions)
- [3] [CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://arxiv.org/abs/2006.06244)
