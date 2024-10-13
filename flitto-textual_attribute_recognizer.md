# 1. 문제 정의
- 기존 이미지 번역 결과물에 있어서, 번역 자체의 품질과 무관하게 이미지의 시각적 품질이 'N'사의 서비스 대비 매우 떨어진다고 판단했습니다.
    |원본 이미지|기존 이미지 번역|||
    |-|-|-|-|
    |한국어|영어|일본어|중국어 간체|
    |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_original.jpg" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_ko_en_as_is.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_ko_ja_as_is.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_ko_zh-CN_as_is.png" width="150">|
 - 문제점을 정리하면 다음과 같습니다.
    - 너무 작거나 큰 폰트 크기:
        - 단순히 바운딩 박스의 크기에 따라 폰트 크기를 계산하는 알고리즘을 사용했습니다.
        - 이로 인해 품질이 바운딩 박스 생성 작업자의 역량에 크게 좌우된다는 문제점이 있었습니다.
    - 잘못된 텍스트 방향:
        - 텍스트 방향은 가로쓰기 또는 세로쓰기를 말합니다.
        - 바운딩 박스의 가로 길이와 세로 길이를 서로 비교해 텍스트 방향을 계산하는 아주 단순한 알고리즘을 사용했습니다.
        - 대부분의 경우에 알고리즘이 잘 작동했지만 그렇지 못했을 경우 사용자 경험을 크게 저해했으며 품질이 바운딩 박스 생성 작업자의 역량에 크게 좌우된다는 점이 문제였습니다.
    - 모두 왼쪽으로만 정렬된 텍스트:
        - 대부분의 경우 가로쓰기 텍스트는 왼쪽 정렬, 세로쓰기 텍스트는 위쪽 정렬이지만, 일부 이미지에서 모든 텍스트에 대해서 왼쪽 정렬 또는 위쪽 정렬만을 사용하면 원본 이미지의 느낌에서 크게 벗어났습니다.
    - 아무 곳에나 삽입된 줄바꿈:
        - 특히 띄어쓰기가 없는 언어 (일본어, 중국어 등)의 경우 하나의 의미 덩어리가 두 줄에 걸쳐 렌더링되는 경우가 많았습니다. 이는 각 언어의 사용 규칙에도 벗어나므로 사용자 경험을 저해했습니다.
    - 단조로운 텍스트 및 텍스트 테두리 색상:
        - 어떤 배경에서도 텍스트의 가독성을 보장하기 위해서 텍스트 및 텍스트 테두리의 색상을 각각 검은색과 흰색 또는 흰색과 검은색으로 했습니다.
        - 원본 이미지에서 다양한 색상이 사용된 경우, 이미지 번역 결과물의 단조로움은 더욱 대비됐습니다.
    - 모든 텍스트에 삽입된 텍스트 테두리:
        - 어떤 배경에서도 텍스트의 가독성을 보장하기 위해 모든 텍스트에 텍스트 테두리를 설정했습니다.
        - 원본 이미지에서 대부분의 텍스트에는 텍스트 테두리가 없으므로, 결과물이 원본 이미지의 느낌에서 크게 벗어나는 문제점이 있었습니다.

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
- 'text region score map'에 connected component labeling (CCL), watershed 등을 활용하여 서로 다른 문자를 구분하는 'text region segmentation map'을 생성합니다.
    |Text region segmentation map|
    |-|
    |<img src="https://user-images.githubusercontent.com/67457712/235050300-e4dff000-f476-485f-8cfd-28799b24e9f8.jpg" width="600">|
    |<img src="https://user-images.githubusercontent.com/67457712/235050726-ab8d3a2f-75cd-4637-8e02-f2105ba41fff.jpg" width="200">|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/06d1994c-c9a8-495a-a634-27d21b06a2e8" width="600">|
- 'text region segmentation map'의 각 레이블의 높이를 구하고 이로부터 통계적인 방법을 통해 바운딩 박스별로 하나의 폰트 크기를 추출합니다.
- 실제로는 하나의 바운딩 박스 안에서 다양한 폰트 크기가 존재할 수도 있지만 모두 동일하다고 가정합니다.
- 추가적으로, 텍스트가 서로 겹쳐 렌더링될 경우에 폰트 크기를 자동으로 저장해서 이를 방지하는 알고리즘을 도입했습니다.
    |텍스트간 겹침 발생|텍스트간 겹침 제거|
    |-|-|
    |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/701_2471_collisions.jpg" width="300">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/701_2471_collisions_decreased.jpg" width="300">|

## 2) 텍스트 방향
- 'text region score map'을 사용해 각 문자의 중심 (Pseudo Character Center (PCC) [3])을 추출합니다.
    |Pseudo Character Centers (PCCs)|
    |-|
    |<img src="https://user-images.githubusercontent.com/67457712/235050297-a43a5b3c-fdb1-41ad-a30c-30e0f2778910.jpg" width="600">|
- 각 PCC에 대해서 가장 가까운 다른 PCC를 찾습니다. 두 PCC간의 x축 방향과 y축 방향 각각에 대한 거리를 구해서 x축 방향의 거리가 y축 방향의 거리보다 더 멀면 가로쓰기라고 판단하고 y축 방향의 거리가 x축 방향의 거리보다 멀면 세로쓰기라고 판단합니다.
- 이때 가장 가까운 다른 PCC와의 거리가 폰트 크기와 비교해 너무 작다면, 바로 그 다음으로 가까운 PCC를 찾습니다. 이는 동일한 하나의 문자에 대해서 다수의 PCC가 추출되는 에러를 보완하기 위함입니다.

## 3) 텍스트 정렬
- 규칙 기반 방법론으로는 추출이 불가능하다고 판단했습니다 [2]. 또한 각 텍스트만 보고는 알 수 없고 이미지 전체를 봐야만 높은 정확도로 판단할 수 있다고 보았습니다.
- 이에 4개의 클래스 ('none', 'left', 'center', 'right')에 대한 semantic segmentation 문제로 접근했습니다. 세로쓰기 텍스트는 데이터의 수가 매우 부족했기 때문에 전부 위쪽 정렬을 적용하고 가로쓰기 텍스트에 한해서만 모델을 통해 텍스트 정렬을 추출하도록 했습니다.
    |정답 데이터|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/1d4e8384-926c-4de0-bde4-c8517312bdf3" width="600">|
    |빨간색: 'left', 초록색: 'center', 파란색: 'right'|
- 모델의 출력에서 바운딩 박스별로 가장 많은 픽셀 수를 차지하는 클래스를 텍스트 정렬로 추출합니다.
    |원본 이미지|As-is|To-be|
    |-|-|-|
    |<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/125_257-ori.png" width="600">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/125_257-as_is.png" width="600">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/resources/125_257-to_be.png" width="600">|
    <!-- |<img src="https://github.com/KimRass/KimRass/assets/67457712/d27aba57-f3ba-405e-a3f2-157b6e0fa55d" width="600">| -->

## 4) 텍스트 줄바꿈
- 주어진 텍스트를 렌더링할 때 어디에 줄바꿈을 삽입했을 때 폰트 크기가 가장 원본에 가까워질 지를 계산합니다.
- 언어별로 줄의 시작 또는 끝에 올 수 없는 문자 등에 관한 규칙이 있는데 [5], 이에 위배되지 않도록 정교한 문자열 처리 알고리즘을 개발했습니다.

### (1) 띄어쓰기가 있는 언어
- 서비스를 지원하는 언어 중 한국어, 영어, 베트남어, 인도네시아어, 말레이어, 스페인어가 이에 해당합니다.
- 어절과 어절 사이에만 줄바꿈을 삽입할지 여부가 결정됩니다.
- 한국어 예시:
   |구분|줄 수|한 줄당<br>문자의<br>최대 개수|
   |-|-|-|
   |저희 업소에서는 남은 음식물을 재활용하지 않습니다.|1|28|
   |저희 업소에서는 남은 음식물을 재활용하지<br>않습니다.|2|22|
   |저희 업소에서는 남은 음식물을<br>재활용하지 않습니다.|2|16|
   |저희 업소에서는 남은<br>음식물을 재활용하지<br>않습니다.|3|11|
   |저희 업소에서는<br>남은 음식물을<br>재활용하지<br>않습니다.|4|8|
   |저희<br>업소에서는<br>남은 음식물을<br>재활용하지<br>않습니다.|5|7|
- 영어의 경우, `pyphen` 라이브러리를 통해 한 단어를 분절하여 사이에 줄바꿈을 삽입하는 end-of-line hyphenation을 고려합니다.

### (2) 띄어쓰기가 없는 언어
- 서비스를 지원하는 언어 중 일본어, 중국어, 중국어, 태국어가 이에 해당합니다.
- 품사 분석에 기반하여 의미 단위로 텍스트를 분리하고 줄의 시작 또는 끝에 올 수 없는 문자를 고려하여 줄바꿈합니다. 품사 분석을 위해 일본어는 `fugashi`, 중국어는 `budoux`, 태국어는 `pythainlp` 라이브러리를 각각 사용합니다.
- 일본어 예시:
   |구분|줄 수|한 줄당<br>문자의<br>최대 개수|
   |-|-|-|
   |当店では残飯を再活用しません。|1|15|
   |当店では残飯を再活用<br>しません。|2|10|
   |当店では残飯を再<br>活用しません。|2|8|
   |当店では残飯を<br>再活用<br>しません。|3|7|
   |当店では<br>残飯を再活用<br>しません。|3|6|
   |当店では<br>残飯を再<br>活用<br>しません。|4|5|
- 중국어 예시:
   |구분|줄 수|한 줄당<br>문자의<br>최대 개수|
   |-|-|-|
   |本店绝不使用剩菜。|1|9|
   |本店绝不使用剩<br>菜。|2|7|
   |本店绝不使用<br>剩菜。|2|6|
   |本店绝不<br>使用剩菜。|2|5|
   |本店绝<br>不使用<br>剩菜。|3|3|
- 위 경우 각각에 대해, 추출된 폰트 크기로 바운딩 박스를 벗어나지 않도록 렌더링을 시도하고 바운딩 박스를 벗어난다면 벗어나지 않도록 폰트 크기를 줄입니다.
- 가장 폰트 크기가 클 수 있는 줄바꿈을 최종적으로 선택하고 같은 폰트 크기에 대해서는 줄바꿈 횟수가 가장 적은 것을 선택합니다.

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
- 검은색과 흰색 중 텍스트 색상의 보색과 이루는 명암비가 더 큰 쪽을 선택하고 이를 CIELAB 색 공간 상에서 텍스트 색상과 적절히 보간한 값을 텍스트 테두리 색상으로 지정합니다.
- 이렇게 하는 이유는 가독성을 높이기 위해 텍스트 색상의 보색을 텍스트 테두리 색상으로 사용함과 동시에 텍스트 테두리 색상이 너무 튀지 않도록 하기 위한 휴리스틱한 알고리즘입니다.

## 7) 텍스트 테두리 두께
- 텍스트 테두리를 적용하지 않았을 때 그 텍스트를 사람이 읽을 수 없을 때에만 적용한다는 규칙을 만들고 싶었습니다.
- 폰트 크기, 텍스트 정렬, 텍스트 줄바꿈이 결정되면 각 문자가 어디에 렌더링될 지가 결정됩니다.
- 각 문자에서 텍스트를 둘러싼 영역의 픽셀마다 배경 색상을 분석합니다. 각각의 배경 색상과 앞서 정해진 텍스트 색상 사이의 명암비를 계산합니다. 이 값이 일정한 값 미만이라면 그 두 색상 조합은 명암이 좋지 않은 것으로 판단합니다 [4].
    |텍스트를 둘러싼 영역|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/7dbc77da-cdeb-48d0-bc0e-78ea0bd9024b" width="200">|
    <!-- |흰색 영역의 픽셀마다 배경 색상을 분석.| -->
- 텍스트를 둘러싼 영역의 전체 픽셀 수 대비 명암이 좋지 않은 픽셀의 수를 계산합니다. 이 값을 각 문자의 '가독성'이라고 할 때, 바운딩 박스 내의 문자 중 가독성이 좋지 않은 문자가 하나라도 있다면 그 텍스트에는 텍스트 테두리를 적용합니다.
- 텍스트 테두리의 두께는 이미지 해상도 대비 폰트 크기를 통해 계산됩니다.
    <!-- |Text readibility|
    |-|
    |<img src="https://github.com/KimRass/KimRass/assets/67457712/672349f4-3781-4031-ba7c-046a7779c189" width="900">| -->

## 8) 텍스트 속성 외
- 문자열 처리:
    - 물음표 앞의 공백, 가운뎃점 앞뒤의 공백 등 특수문자가 잘못 사용된 경우를 수정합니다.
    - 사용자가 텍스트의 시작과 끝을 쉽게 인식할 수 있도록 여러 줄에 걸친 텍스트에 엔 대시를 삽입하는 방안 실험.
- 렌더링:
    - CSS에서 텍스트 테두리를 표현하기 위해 `text-stroke` 속성이 아닌 `text-shadow` 속성을 사용하도록 변경.
    - 일본어, 중국어, 태국어에 대해서 `text-shadow` 속성을 지원하고 심미성이 높은 폰트로 변경.
    - 파이썬으로 프론트 엔드와 동일한 텍스트 렌더링 결과물을 얻을 수 있도록 'Pillow' 라이브러리로 텍스트 렌더링 엔진 개발.

# 3. 성과

## 1) 예시
|원본 이미지|개선된 이미지 번역|||
|-|-|-|-|
|한국어|영어|일본어|중국어 간체|
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_original.jpg" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_ko_en_to_be.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_ko_ja_to_be.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1283_3030_ko_zh-CN_to_be.png" width="150">|
|한국어|영어|일본어|중국어|
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_ko_original.jpg" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_en_after.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_ja_after.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1499_5721_zh-cn_after.png" width="150">|
|한국어|영어|일본어|중국어|
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_ko_original.jpg" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_en_after.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_ja_after.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/new/1508_5739_zh-cn_after.png" width="150">|
|중국어 번체|영어|||
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1816_7274_original.jpg" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1816_7274_zh-TW_en_to_be.png" width="150">|||
|일본어|영어|한국어|중국어 간체|
|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1380_5047_original.jpg" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1380_5047_ja_en_to_be.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1380_5047_ja_ko_to_be.png" width="150">|<img src="https://raw.githubusercontent.com/KimRass/KimRass/refs/heads/main/Flitto/Place-Translation/Textual-Attribute-Recognizer/examples/1380_5047_ja_zh-CN_to_be.png" width="150">|

## 2) 개선 사항
<table>
    <thead>
        <tr>
            <th>텍스트 속성</th>
            <th>As-is</th>
            <th>To-be</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>폰트 크기</td>
            <td>원본 폰트 크기와 무관하게 바운딩 박스에 의해서만 결정.</td>
            <td>• 충분한 텍스트 공간이 확보된다면 바운딩 박스와 무관.<br>• 원본 폰트 크기보다 작거나 같고 바운딩 박스를 최대한 활용할 수 있는 폰트 크기 추출.</td>
        </tr>
        <tr>
            <td>텍스트 방향</td>
            <td>바운딩 박스에 의해서만 결정되며 정확성이 떨어짐.</td>
            <td>정확성이 높음.</td>
        </tr>
        <tr>
            <td>텍스트 정렬</td>
            <td>모든 가로쓰기 텍스트에 왼쪽 정렬 적용.</td>
            <td>가로쓰기 텍스트에 대하여 높은 정확도로 텍스트 정렬 추출.</td>
        </tr>
        <tr>
            <td>텍스트 줄바꿈</td>
            <td>언어별 사용 규칙과 무관하게 줄바꿈이 삽입되는 경우가 다수.</td>
            <td>언어별 사용 규칙과 텍스트의 의미를 고려하여 적절한 위치에 줄바꿈 삽입.</td>
        </tr>
        <tr>
            <td>텍스트 색상</td>
            <td>검은색 또는 흰색으로 단조로움.</td>
            <td>원본 이미지의 텍스트 색상를 반영하여 다채롭고 생동감 있는 느낌 전달.</td>
        </tr>
        <tr>
            <td>텍스트 테두리 색상</td>
            <td>검은색 또는 흰색 텍스트의 보색을 사용하여 텍스트 테두리가 지나치게 강조됨.</td>
            <td>추출된 텍스트 색상과의 적절한 보간을 통해 강조가 완화됨.</td>
        </tr>
        <tr>
            <td rowspan="2">텍스트 테두리 두께</td>
            <td>모든 텍스트에 텍스트 테두리가 적용되어 가독성은 보장되나 심미성이 떨어짐.</td>
            <td>가독성을 위해 필요한 경우에만 텍스트 테두리 두께를 적용하여 심미성 향상.</td>
        </tr>
        <tr>
            <td>CSS의 `text-stroke` 속성을 사용하여 텍스트가 빛나는 듯한 효과.</td>
            <td>CSS의 `text-shadow` 속성을 사용하여 텍스트가 빛나지 않아 가독성 향상.</td>
        </tr>
    </tbody>
</table>

# 4. References
- [1] [Character Region Awareness for Text Detection](https://github.com/KimRass/CRAFT/blob/main/papers/character_region_awareness_for_text_detection.pdf)
- [2] ['더' 잘 읽히고 자연스러운 이미지 번역을 위해 (파파고 텍스트 렌더링 개발기)](https://deview.kr/2023/sessions)
- [3] [CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://arxiv.org/abs/2006.06244)
- [4] Web Content Accesibility Guidelines (WCAG)
- [5] [Line breaking rules in East Asian languages](https://en.wikipedia.org/wiki/Line_breaking_rules_in_East_Asian_languages)
