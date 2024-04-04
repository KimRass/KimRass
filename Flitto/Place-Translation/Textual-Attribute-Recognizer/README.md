- 이미지 상의 각 텍스트로부터 속성을 추출하여 이를 번역문 렌더링 시 활용함으로써 이미지 번역 품질을 향상시키는 모델입니다.
- Textual attributes:
    - [Font size](#1-font-size)
    - [Writing direction](#2-writing-direction)
    - [Text alignment](#3-text-alignment)
    - [Text line breaking](#4-text-line-breaking)
    - [Text color](#5-text-color)
    - [Text border](#6-text-border)
    - [Text border color](#7-text-border-color)

# 1. Font Size
- Original image
    - <img src="https://user-images.githubusercontent.com/67457712/235050769-a7970aa6-2b99-4de0-a20b-1bf301ad7d38.jpg" width="600">
- Bounding box annotation
    - <img src="https://user-images.githubusercontent.com/67457712/235050291-34d76d66-cfdc-47fe-a86a-8ea177eda211.jpg" width="600">
- 'CRAFT' scene text detection model [1]을 사용해 'text region score map'을 추출합니다. 어떤 픽셀이 빨간색에 가까울수록 모델이 그 픽셀을 텍스트의 중심이라고 확신함을 나타냅니다.
    - <img src="https://user-images.githubusercontent.com/67457712/235050295-8b42ccdf-976c-4dc3-a0f8-3ceffee8bf7a.jpg" width="600">
- connected component labeling, watershed 등을 사용해 각 문자를 서로 다른 Class로 구분하는 'text region segmentation map'을 생성합니다. 각 class가 정확히 하나의 문자를 나타내지는 못하므로 pseudo characters라고 부르겠습니다.
    - <img src="https://user-images.githubusercontent.com/67457712/235050300-e4dff000-f476-485f-8cfd-28799b24e9f8.jpg" width="600">
    - <img src="https://user-images.githubusercontent.com/67457712/235050726-ab8d3a2f-75cd-4637-8e02-f2105ba41fff.jpg" width="200">
    - <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/105417680/247047557-593c6476-042a-47cf-9a99-cf093581cf0e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240404T001355Z&X-Amz-Expires=300&X-Amz-Signature=ddee9e975d8e9b23c28afd2a80b8b7719e1357985addeeaf08f83f47aa542826&X-Amz-SignedHeaders=host&actor_id=67457712&key_id=0&repo_id=291084893" width="600">
    - (매우 많은 수의 label이 생성되지만 이해를 돕기 위해 26개의 색상을 사용하여 단순화했습니다.)
- 각 pseudo character의 크기에 일정한 값을 곱해 이를 통해 해당 문자의 font size로 구합니다. 실제로는 하나의 bounding box에서 다양한 font size가 존재할 수도 있지만 모두 동일하다고 가정하여 통계적인 방법을 통해 하나의 font size를 추출합니다.

# 2. Writing Direction
- text region segmentation map으로부터 pseudo character centers (PCCs) [3]를 추출합니다. 
    - <img src="https://user-images.githubusercontent.com/67457712/235050297-a43a5b3c-fdb1-41ad-a30c-30e0f2778910.jpg" width="600">
- 각 PCC에 대해서 가장 가까운 다른 PCC를 찾습니다. 두 PCCs간의 x축 방향과 y축 방향 각각에 대한 거리를 구해서 x축 방향의 거리가 y축 방향의 거리보다 더 멀면 가로쓰기라고 판단하고 y축 방향의 거리가 x축 방향의 거리보다 멀면 세로쓰기라고 판단합니다.
- 이때 가장 가까운 다른 PCC와의 거리가 font size와 비교해 너무 작다면, 바로 그 다음으로 가까운 PCC를 찾습니다. 이는 동일한 하나의 문자에 대해서 다수의 PCCs가 추출되는 경우에 이를 보완하기 위함입니다.

# 3. Text Alignment
- rule-based approach로는 추출이 불가능하다고 판단했습니다 [2]. 또한 각 bounding box의 이미지 영역만을 보고는 알 수 없으며 이미지의 전체적인 visual features를 고려하여야만 높은 정확도로 판단할 수 있다고 보았습니다.
- 이에 4개의 class ('none', 'left', 'center', 'right')에 대한 semantic segmentation 문제로 접근했습니다. 가로쓰기 텍스트에 대해서는 'left', 'center', 'right' 중 하나로, 세로쓰기 텍스트에 대해서는 'none'으로 레이블링을 했습니다. 즉 가로쓰기 텍스트에 한해서만 모델이 text alignment를 예측하도록 학습시켰습니다.
    - <img src="https://private-user-images.githubusercontent.com/67457712/287611407-089e7647-688a-4d4d-8646-d417ff7ffd51.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yODc2MTE0MDctMDg5ZTc2NDctNjg4YS00ZDRkLTg2NDYtZDQxN2ZmN2ZmZDUxLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI0ZGUxNDlmYzZhNDdjNjUxZjc5ZDg0ZGZkYWQzNWZiOGI1NDVlNjEzNmQ3NzYwMDIxYzE4OTEzMWY3YjZkZTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.GMRDd0y18FbZ20ZM5y0bVAKDX9wn76ILmaHzarLLvaA" width="500">
    - 빨간색: 'left', 초록색: 'center', 파란색: 'right', 그 외: 'none'
- 모델이 내놓은 결과에 대해서, 각 bounding box가 차지하는 영역에서 가장 많은 픽셀 수를 차지하는 class를 통해 text alignment를 예측합니다.
<!-- - 1,000장의 이미지, 19,523개의 bounding box에 대해 학습한 결과 mIoU 0.7341 -->
- Examples
    - <img src="https://private-user-images.githubusercontent.com/67457712/287599160-f0d354a8-348d-484f-95e4-816c037108b0.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yODc1OTkxNjAtZjBkMzU0YTgtMzQ4ZC00ODRmLTk1ZTQtODE2YzAzNzEwOGIwLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM5YzQ0NjNlOGM1MWUwNDEyMWY1ZjBlZmM1ZWVlY2IzYjRhMTNmODExM2M0ZjNmNmE5NzFmYjM0NGZiMTRiOTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.i58VwVVu4WxyyQuUvB0Evn9T9rhpzH11mrZJ4xkO_Og" width="800">
    - <img src="https://private-user-images.githubusercontent.com/67457712/287599185-edf361f2-6343-4767-a601-7e5ad091a28b.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yODc1OTkxODUtZWRmMzYxZjItNjM0My00NzY3LWE2MDEtN2U1YWQwOTFhMjhiLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI1ZDFjODFjMzQzMzdmZjAyYjhkYjk1ZTE3OTIyNDFlZTNjMjM0Mzc2ODMzNGZkMzQ1ZDVkMTY2NzYxMTVlY2YmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.cGELW4D6jiZDpdJLssOdhj7fpIQq0kSyrCBlRoRxPnA" width="800">
    <!-- - <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/67457712/287581756-d2460fbf-9716-4b52-b2c6-0695264a4d33.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240404T000439Z&X-Amz-Expires=300&X-Amz-Signature=9f3bcc9523004783714da96d1b10865e8631f7aba8baad1620dab455a74f92ce&X-Amz-SignedHeaders=host&actor_id=67457712&key_id=0&repo_id=669664726" width="800"> -->

# 4. Text Line Breaking
- 주어진 텍스트를 렌더링할 때 어디에 줄바꿈을 삽입했을 때 font size가 가장 원본에 가까워질 지를 계산합니다.
<!-- - 주어진 텍스트를 렌더링할 때 한 줄로 할 수도 있고 중간에 줄바꿈을 삽입하여 두 줄 이상으로 할 수도 있을 것입니다. 수많은 경우 중 최적을 찾는 알고리즘입니다. -->
## 1) 띄어쓰기가 있는 언어 (ko, en, vi, id, ms, es)
- 어절과 어절 사이에만 줄바꿈을 삽입할지 여부가 결정됩니다.
- 한국어 예시
    | Case | Number of lines | Maximum number<br>of characters<br>on a single line |
    |------|---|---|
    | 저희 업소에서는 남은 음식물을 재활용하지 않습니다. | 1 | 28 |
    | 저희 업소에서는 남은 음식물을 재활용하지<br>않습니다. | 2 | 22 |
    | 저희 업소에서는 남은 음식물을<br>재활용하지 않습니다. | 2 | 16 |
    | 저희 업소에서는 남은<br>음식물을 재활용하지<br>않습니다. | 3 | 11 |
    | 저희 업소에서는<br>남은 음식물을<br>재활용하지<br>않습니다. | 4 | 8 |
    | 저희<br>업소에서는<br>남은 음식물을<br>재활용하지<br>않습니다. | 5 | 7 |
## 2) 띄어쓰기가 없는 언어 (ja, zh-cn, zh-tw, th)
- part-of-speech tagging에 기반하여 의미 단위로 텍스트를 분리하고 줄의 시작 또는 끝에 올 수 없는 문자를 고려하여 줄바꿈합니다.
- 일본어 예시
    | Case | Number of lines | Maximum number<br>of characters<br>on a single line |
    |------|---|---|
    | 当店では残飯を再活用しません。 | 1 | 15 |
    | 当店では残飯を再活用<br>しません。 | 2 | 10 |
    | 当店では残飯を再<br>活用しません。 | 2 | 8 |
    | 当店では残飯を<br>再活用<br>しません。 | 3 | 7 |
    | 当店では<br>残飯を再活用<br>しません。 | 3 | 6 |
    | 当店では<br>残飯を再<br>活用<br>しません。 | 4 | 5 |
- 중국어 예시
    | Case | Number of lines | Maximum number<br>of characters<br>on a single line |
    |------|---|---|
    | 本店绝不使用剩菜。 | 1 | 9 |
    | 本店绝不使用剩<br>菜。 | 2 | 7 |
    | 本店绝不使用<br>剩菜。 | 2 | 6 |
    | 本店绝不<br>使用剩菜。 | 2 | 5 |
    | 本店绝<br>不使用<br>剩菜。 | 3 | 3 |
- 위 경우 각각에 대해, 추출된 font size로 bounding box를 벗어나지 않도록 렌더링을 시도하고 bounding box를 벗어난다면 벗어나지 않도록 font size를 줄입니다.
- 가장 font size가 클 수 있는 줄바꿈을 최종적으로 선택하고 같은 font size에 대해서는 줄바꿈 횟수가 가장 적은 것을 선택합니다. 

# 5. Text Color
- Scene text removal
    - <img src="https://private-user-images.githubusercontent.com/105417680/247048146-d80b176f-f0a3-4ffa-9e45-83b575dcf278.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDQ4MTQ2LWQ4MGIxNzZmLWYwYTMtNGZmYS05ZTQ1LTgzYjU3NWRjZjI3OC5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jOWM3NjllMmM4NmFlNTYzMDhlMjZiOTkyNjc1ODNmYmIwYjU2MTJlOTNiMDdmODM4MzlhZjVhMmRiNmM2OWVmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.XVCWqz5uOI0qoYxZItHma-em1V3HOFjq-egn3uDOcts" width="600">
- Pixel-wise image difference
    - 원본 이미지와 텍스트가 제거된 이미지 사이의 픽셀 단위의 차이를 구합니다.
    - <img src="https://private-user-images.githubusercontent.com/105417680/247062590-2127ef36-4b70-4b3a-938a-b90d693e89a5.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDYyNTkwLTIxMjdlZjM2LTRiNzAtNGIzYS05MzhhLWI5MGQ2OTNlODlhNS5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jOTcxMTg5YTcxZmIzNmRiYjFjNTZjZTdjZWY1MTU2OWYyMzNlZjdkZDFkMmRhNmM1ZTNmNDQwNjg4ZjQ0ZDFiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.gHR8iRas8_mGQSFvtO_lrM4tlnB7TrKoCimc38sX864" width="600">
- 이를 바탕으로 mask를 생성합니다 [2].
    - <img src="https://private-user-images.githubusercontent.com/105417680/247062575-8fd85555-6a72-4a2e-a336-bc597f75708c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDYyNTc1LThmZDg1NTU1LTZhNzItNGEyZS1hMzM2LWJjNTk3Zjc1NzA4Yy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01ZWVmNjFkYTMxZmIzMDE0ZWE4OWJjZjk3ODk1OWI2NzZjNzU2YTUxODA5ZGQxMDVhZjI5MDNmNTA5NTlmNjI5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.ulWSGyq95sKtmP0HytJDmnEDQZbHhc7V37WVy1nFsI0" width="600">
- Color extraction
    - 각 bounding box에 대해서 mask에 해당하는 영역에 대해서만 원본 이미지로부터 픽셀별 색깔을 조사합니다. 이때 비슷하지만 서로 조금씩 다른 색깔은 많은 픽셀 수를 갖는 색깔 쪽으로 통합해 나갑니다.
    - 색깔의 수가 3개가 될 때까지 통합합니다. 이 3개의 색깔 중에서 가장 많은 픽셀 수를 갖는 `(7, 145, 58)`을 text color로 정합니다.
        - Original image
            - <img src="https://user-images.githubusercontent.com/67457712/235061924-e372749d-fa8f-4db2-a655-ad5210ff0c88.jpg" width="400">
        - Pixel-wise image difference
            - <img src="https://user-images.githubusercontent.com/67457712/235061927-f1e4e5ff-fcc4-4640-9817-4d14ba467e28.jpg" width="400">
        - Text color candidates
            - <img src="https://user-images.githubusercontent.com/67457712/235064154-1ff39941-f237-4639-8735-db7bf116986a.jpg" width="300">

# 6. Text Border
- text border를 적용하지 않았을 때, 사람이 그 텍스트를 읽을 수 없으면 text border를 적용하고 읽을 수 있다면 적용하지 않습니다.
- font size, text alignment, text line breaking이 결정되면 각 문자가 어디에 렌더링될 지가 결정됩니다. 각 문자에 대해서, 추출된 text color와 text stroke를 둘러싼 영역의 픽셀 하나하나 사이의 contrast ratio를 계산합니다. 이 값이 일정한 값 미만이라면 그 두 색깔 조합은 가독성이 좋지 않은 것으로 판단합니다.
- 각 문자에 대해서, text stroke를 둘러싼 영역의 전체 픽셀 수와 비교하여 가독성이 좋지 않은 픽셀의 수를 계산합니다. 이 값은 그 문자의 영역 중 읽을 수 있는 부분의 비율을 나타낸다고 볼 수 있습니다.
    - <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/105417680/245984852-13643982-e41d-412a-9164-0e6d79ab4b47.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240404T000655Z&X-Amz-Expires=300&X-Amz-Signature=bcdc565e1050ef550627e71f1e99b99d46e7d70fe5b29eb865fecc1cad20788d&X-Amz-SignedHeaders=host&actor_id=67457712&key_id=0&repo_id=291084893" width="200">
- 이 값을 그 문자의 readability라고 할 때, readability가 어떤 값 미만인 문자가 1개라도 존재할 경우 (1개의 문자라도 가독성이 좋지 않은 것이 있다면) 그 텍스트에는 text border를 적용합니다.
    - <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/67457712/287637459-797a41a5-9743-4f59-80d7-2b6501d07234.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240404T000656Z&X-Amz-Expires=300&X-Amz-Signature=de48e8652e8e24cc47db530c0579452fb5825af3c0caf86ecbe07606ed700ed1&X-Amz-SignedHeaders=host&actor_id=67457712&key_id=0&repo_id=669664726" width="900">

# 7. Text Border Color
- 검은색과 하얀색 중, 추출된 text color의 보색과 contrast ratio가 더 큰 쪽을 선택합니다. 이것을 A라 합니다.
- CIELAB 색 공간 상에서 A와, 추출된 text color의 보색과의 linear interpolation을 수행한 값을 text border color로 지정합니다. 이렇게 하는 이유는 가독성을 높이기 위해 text border의 보색을 text border로서 사용함과 동시에 text border color가 너무 돋보이지 않도록 하기 위함입니다.

# 8. Rendering Quality Improvement
## 1) Inter-text Conflict Prevention
- bounding box annotation 과정에서 발생한 bounding box간의 겹침 (human error)로 인해 발생하는 텍스트간 충돌을 제거하는 알고리즘입니다.

| Original image and bounding box annotation |
|:-|
| <img src="https://private-user-images.githubusercontent.com/67457712/293923830-22b69cdc-dc5a-4cd7-b7da-8133a2374e46.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5MjM4MzAtMjJiNjljZGMtZGM1YS00Y2Q3LWI3ZGEtODEzM2EyMzc0ZTQ2LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM3MTU3NGQ5NWFlYmYyMjYxY2MyNWJjZTY2YWI4MDRjODRiMTU0MWYxZDc1MjAxOThhMTlhNGQwZTZmOTNhYmEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.aRvV2aompVwoLHJkzov3OV4uKCtSO_VV7sD2Mzy-oNo" width="600"> |

| Before and after performing inter-text conflict prevention |
|:-|
| <img src="https://private-user-images.githubusercontent.com/67457712/293923993-428021fd-a8a1-4863-9729-844023fc8b72.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5MjM5OTMtNDI4MDIxZmQtYThhMS00ODYzLTk3MjktODQ0MDIzZmM4YjcyLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRjYTM2OWViY2RkZjQ0OWJlMWU2ODAzNTMxMzVmODdjNmFkOWEwNWQ1NTQ0MTdjYzhkZTk3NGU0YzU5Njk1NjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.LpN-qlAUJUIXTgN2kQMFXH45PieFOAIAfn1ak18bBZ0" width="600"> |

## 2) End-of-line Hyphenation
- 바운딩 박스간의 거리가 가까울 때 각 텍스트가 어디서 시작하고 어디서 끝나는 지 알기 어려운 경우가 많습니다. 이에 대한 해결방안으로서 연구 중입니다.
- 'End-of-line' mode: 텍스트가 2줄 이상일 경우 마지막 줄을 제외하고 각 줄의 마지막에 En dash 삽입
- 'Start-of-line' mode: 텍스트가 2줄 이상일 경우 마지막 첫 번째 줄을 제외하고 각 줄의 처음에 En dash 삽입

| Original text, 'End-of-line' mode and 'Start-of-line' mode |
|:-|
| <img src="https://github.com/flitto/express-param/assets/67457712/a0c8742e-0c90-40cb-87de-9d3b14b2c623" width="900"> |

# 9. Examples
## 1) '화정 by 카쿠시타'
| Original image |
|:-|
| <img src="https://private-user-images.githubusercontent.com/67457712/293941470-c4b2e456-cba5-45a3-9962-fa50c143ee32.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5NDE0NzAtYzRiMmU0NTYtY2JhNS00NWEzLTk5NjItZmE1MGMxNDNlZTMyLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTE2NDNkNjcwYmVkMDA1YWViMzQ0ZmFiYzgyZTE0NGQ4YjNmMmUxYTMwNzFhNzc3ODQ5NjZlYjEzZDQzNmJkNTgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.IXLiZcmWO5JJOWLKYW6EAN5Zw8TJL1fKPIeAwCLDP_Q" width="250"> |

| Textual attribute recogition (en, ja and zh-cn) |
|:-|
| <img src="https://private-user-images.githubusercontent.com/67457712/293941865-6db3e213-162b-4ef5-921c-9f45f1314df6.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5NDE4NjUtNmRiM2UyMTMtMTYyYi00ZWY1LTkyMWMtOWY0NWYxMzE0ZGY2LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiMmNmOGVhMjYyZGQ5MTllOWIyNjZlZTdjMGIyOTFmOWRlMzc4M2U0ZjdiMGZkYjVmZjJlMjlhZGVlYTc0NGYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0._MwE4rhD0hFS5WsoJhHBjr89zswqC1pbpUEshYqm-SI" width="750"> |
## 2) '어반 웍 스테이션'
| Original image |
|:-|
| <img src="https://private-user-images.githubusercontent.com/67457712/293943994-8f6dc2bb-28ca-4a59-a11f-eb36bd7f5715.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5NDM5OTQtOGY2ZGMyYmItMjhjYS00YTU5LWExMWYtZWIzNmJkN2Y1NzE1LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTkzZGRjZjVlNWQ5MTllN2Y4MDQ2ZjZjY2RiMzQwNjU5YmQ2NjM4NDI1ODA2ZTc0M2M1Y2NmMzAwNjJkYWYzMTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0._jNd4R4AVRnX36M1nMk-xVw9rtCtdfDbtZdhTnK1otc" width="200"> |

| Textual attribute recogition (en, ja and zh-cn) |
|:-|
| <img src="https://private-user-images.githubusercontent.com/67457712/293944572-e70ed722-eec4-434d-8529-2d8fdfccd803.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5NDQ1NzItZTcwZWQ3MjItZWVjNC00MzRkLTg1MjktMmQ4ZmRmY2NkODAzLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRkOTRiMTYzOTBlYjYwYzVhNTZjY2NiMmMyNDYyYzNiMTlkMDU4Y2UwYzQ1OWZkZjljZGM4ZjViYTRhNmI5MjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.kG5mGjjwWBl9ITpdbOp8-ddjkP46YQQnAWxtBQQ5Wtc" width="600"> |

# 10. Text Region Recognition
- 기존에는 텍스트가 렌더링될 수 있는 영역을 bounding box가 차지하는 영역으로만 한정했습니다. 그러면 한국어를 영어로 번역할 때와 같이 원문보다 번역문이 긴 경우에는 필연적으로 원본보다 작은 font size로 텍스트를 렌더링할 수밖에 없습니다.
- text region recognition은 bounding box가 차지하는 영역을 벗어나 더욱 큰 font size를 가지고 텍스트를 렌더링할 수 있는지 계산하는 알고리즘입니다 [2].
- Original image
    - <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/105417680/247049197-45467259-88e4-482e-9d47-92bdcd98877f.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240404T004054Z&X-Amz-Expires=300&X-Amz-Signature=8281b0b5f8846492d7948dacb311e66a12693200d71fda642c13f802fbf8be31&X-Amz-SignedHeaders=host&actor_id=67457712&key_id=0&repo_id=291084893" width="500">
- Human-annotated bounding boxes
    - <img src="https://private-user-images.githubusercontent.com/105417680/247049193-4b08bdb7-1fdd-413e-998f-3ef70f9e3cac.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDQ5MTkzLTRiMDhiZGI3LTFmZGQtNDEzZS05OThmLTNlZjcwZjllM2NhYy5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05NTExMjg1NDE3YzY5MDJmNzZmMmExYjEzMzhkMzMxOTI4NjhmMGFhNTM0Mzk5ZWM4ZDM3OTY3YmIzMTVmOTM2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.WFs_FmAOZNTpbqK0cbRBINYlyecgW2ugh-ykEtkWD7c" width="500">
- Text-removed image
    - <img src="https://private-user-images.githubusercontent.com/105417680/247049184-32afb8be-9464-4b60-bd1d-0c82f1a0d8fc.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDQ5MTg0LTMyYWZiOGJlLTk0NjQtNGI2MC1iZDFkLTBjODJmMWEwZDhmYy5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yOWIwOWRlNDUxOThiMzlmNDU1NjE3MGQwZGRhOWRlZmRmODU2NGYxYTk2ODY1OTNmZDJlOTg1YzQ5NWQ2M2I1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.d8X9HwzR0iZBNkX_J3kyVwZ5jMh0XCpNrfemvPN8xqQ" width="500">
## Mask generation
- 렌더링할 텍스트가 이미지 상의 주요 오브젝트와 충돌하지 않도록 제한하는 과정이 필요합니다. 이를 위해 3가지 마스크를 사용합니다.
    - Image boundary mask
        - <img src="https://private-user-images.githubusercontent.com/105417680/247051120-13054ce6-eeb3-47ca-9169-6647670ca00f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDUxMTIwLTEzMDU0Y2U2LWVlYjMtNDdjYS05MTY5LTY2NDc2NzBjYTAwZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ODE3MjZhNWY5OGUxZjk1MTRhYTM4NGNkODFhYzVjOTEyNWQ4YTIyNWE0NGRhZDZiNDE3ZTYwNGFlMTQ5N2FiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.AvwfhHPtsLzvAx5-oXvOT4iikyUQ57frkJKdm6PLD4M" width="400">
        - font size를 점진적으로 확대해 나갈 때 텍스트가 이미지 바깥으로 나가지 않도록 하기 위해 사용됩니다.
    - Edge mask [2]
        - <img src="https://private-user-images.githubusercontent.com/105417680/247050649-593277ba-98b0-4daa-bc93-01f07e6e3d2b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDUwNjQ5LTU5MzI3N2JhLTk4YjAtNGRhYS1iYzkzLTAxZjA3ZTZlM2QyYi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kMzRlYzU2NmYzOTA0ZWUzNTJlNTI4Y2I4Y2Y1YjhmZjQ5NDIzYzM5NDBkYWYzNWFiOTZlZWI4MjdjYTkxOGJhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.ZLOh1TZJRlGoYuR--LolCS3bizTGYgWPPzThBfQC6P0" width="400">
        - 텍스트가 이미지에 존재하는 Edge를 넘어가지 않도록 제한하기 위해 사용됩니다.
    - Mask for remaining text
        - <img src="https://private-user-images.githubusercontent.com/105417680/247051838-2ddec590-4911-4eaf-9733-415654c56b2f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDUxODM4LTJkZGVjNTkwLTQ5MTEtNGVhZi05NzMzLTQxNTY1NGM1NmIyZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01ODNhN2FhYmE2YmM3ZDEzZmRjOWE2ZTJiYmMxNDlhM2E1ZjU4NDY5MjI2ZTNjZjY2MjM1YTU4YWQyNGRhNWUwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Z-ZjNwZDiEFX_E5nmTM-d61JAdcDWaJbjwcSghHTHww" width="400">
        - 가격이나 음식점 이름과 같이 '메뉴 번역' 서비스의 정책상 제거하지 않은 텍스트와, 렌더링하고자 하는 텍스트가 서로 겹치지 않도록 하기 위해 사용됩니다.
    - Intermediate mask
        - <img src="https://private-user-images.githubusercontent.com/105417680/247054336-d0973eb5-8549-4c83-b591-c93ec178d954.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii8xMDU0MTc2ODAvMjQ3MDU0MzM2LWQwOTczZWI1LTg1NDktNGM4My1iNTkxLWM5M2VjMTc4ZDk1NC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQwNFQwMDQyMTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wOTg3MTk3YzYyOGU0NDUxNWY0MWNkZmIxMzIwNDc3MmE0MmVhNmJlNTQ2NDYwMzQxNGYwOGM1NzI4YzdmMGVkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.9BFN_KshZES0nLqFIprIUuuZenidixE7KP7zTwc3nzk" width="400">
        - 위 3개의 마스크는 렌더링할 텍스트에 무관하게 변하지 않으므로 빠른 계산을 위해 모두 합쳐 intermediate mask를 생성합니다.
- 이 모든 마스크의 합과 렌더링할 타겟 텍스트가 충돌하지 않는 범위 내에서 조금씩 font size를 증가시킵니다. 이때 원본 font size보다 크게 증가시키지는 않습니다.
- 이때 4번마스크는 렌더링할 타겟 텍스트 자기 자신을 제외한 나머지 텍스트로 합니다. 이 4가지 마스크들의 합으로 새로운 마스크를 정의합니다.
- 추출된 font size의 절반에서 시작해 원본 font size의 0.6배 → 0.7배 → 0.8배 → 0.9배 → 1.0배 순으로 총 6개의 font size를 가지고 텍스트 렌더링을 시도합니다.
- 1번에서 정의한 마스크와 렌더링한 텍스트가 서로 충돌하는지 (겹침이 발생하는지) 확인합니다. 충돌이 발생하지 않으면 해당 font size를 해당 텍스트의 font size로 재정의합니다.
충돌 판별 로직:
    - 텍스트가 길 경우 줄바꿈의 수가 주어졌을 때 가능한 줄바꿈의 경우의 수는 매우 많습니다. 따라서 이 모든 경우를 하나씩 렌더링해보는 것은 매우 많은 계산량을 요구합니다.
    - 텍스트를 공백을 기준으로 서브워드 단위로 분리합니다. 분리된 서브워드들을 하나씩 붙여가면서 원래의 텍스트에 가깝게 조금씩 문자열을 만들어갑니다.
    - 완성되어가는 문자열을 가지고 렌더링을 시도합니다. 충돌이 발생하지 않으면 다음 서브워드를 그대로 이어붙여 다음 문자열을 만들고 충돌이 발생하면 줄바꿈을 중간에 삽입한 체로 다음 서브워드를 이어붙입니다. 충돌이 2회 연속 발생하면 해당 font size로의 확대에 실패한 것이므로 해당 bounding box의 최대 font size는 바로 직전 단계에서 텍스트 렌더링에 성공한 font size가 됩니다.
- Before and after
    - <img src="https://private-user-images.githubusercontent.com/67457712/293927112-e9f879d7-d3f2-4b5e-80dd-0ec665ca8952.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTIxOTE2MzYsIm5iZiI6MTcxMjE5MTMzNiwicGF0aCI6Ii82NzQ1NzcxMi8yOTM5MjcxMTItZTlmODc5ZDctZDNmMi00YjVlLTgwZGQtMGVjNjY1Y2E4OTUyLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDA0VDAwNDIxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNlNjViNzI0YjhmYTZlZjU3ZDQ1ZGFhYTM4MzczMTQ3YjYzZjNhNjg0YzVjYTI1MzA3MmRjODVmY2E5ODA1YzMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.0Wvd6tvK9mJTe5Dev8lIl1tMbxNppBo-xuxSV8x4_UM" width="900">

# 11. Improvements
| Textual attribute | Before | After |
|-|-|-|
| Font size | - bounding box의 가로와 세로 길이에 의해 결정되므로 원본 font size와 크게 달라질 수 있음<br>- 원본 font size보다 커질 수 있음  | - 충분한 텍스트 렌더링 공간이 확보된다면 bounding box의 가로와 세로 길이에 무관<br>- 원본 font size를 넘지 않는 한도 내에서 주어진 boundig box를 최대한 활용하는 font size 추출 |
| Writing direction | bounding box의 가로와 세로 길이의 비율에 의해 결정되므로 틀리게 추출되는 경우가 많음 | text detection model이 잘 감지하지 못하는 텍스트를 제외하고는 높은 인식률
| Text alignment |  |  |
| Text line breaking | - 띄어쓰기가 없는 언어의 경우 하나의 의미를 갖는 텍스트가 두 줄에 걸쳐 렌더링되는 경우가 많음 |  |
| Text color | 검은색 또는 하얀색으로 단조로움 | 원본 이미지의 text color를 반영함으로써 다채롭고 생동감 있는 느낌을 전달 |
| Text border | - 모든 텍스트에 text border가 적용되어 가독성은 보장되나 심미성이 현저히 떨어짐<br>- CSS의 `text-stroke` 속성을 통해 구현되어 글자가 빛나는 듯한 효과 | - 가독성을 위해서 text border가 꼭 필요한 경우에만 적용하므로 심미성 향상<br>- CSS의 `text-stroke` 속성을 통해 구현되어 깔끔함
| Ttext border color | 텍스트가 검은색 일 경우에는 하얀색을, 반대의 경우에는 검은색을 사용함으로써 text border가 지나치게 강조됨 | 추출된 text color와의 interpolation을 통해 text border가 지나치게 강조되는 느낌을 완화

# 12. References
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
