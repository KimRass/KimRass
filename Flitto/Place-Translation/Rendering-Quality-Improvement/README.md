# 1. Rendering Quality Improvement Techniques

## 1) Inter-text Conflict Prevention
- bounding box annotation 과정에서 발생한 bounding box간의 겹침 (human error)로 인해 발생하는 텍스트간 충돌을 제거하는 알고리즘입니다.

| Original image and bounding box annotation |
|:-|
| <img src="https://github.com/KimRass/KimRass/assets/67457712/27e1bfbd-6cdd-4443-b8d5-a18fc4b07e52" width="600"> |

| Before and after performing inter-text conflict prevention |
|:-|
| <img src="https://github.com/KimRass/KimRass/assets/67457712/96937e89-5b73-4e41-8ab2-d0cc5cc7f0ce" width="600"> |

## 2) End-of-line Hyphenation
- 바운딩 박스간의 거리가 가까울 때 각 텍스트가 어디서 시작하고 어디서 끝나는 지 알기 어려운 경우가 많습니다. 이에 대한 해결방안으로서 연구 중입니다.
- 'End-of-line' mode: 텍스트가 2줄 이상일 경우 마지막 줄을 제외하고 각 줄의 마지막에 En dash 삽입
- 'Start-of-line' mode: 텍스트가 2줄 이상일 경우 마지막 첫 번째 줄을 제외하고 각 줄의 처음에 En dash 삽입

| Original text, 'End-of-line' mode and 'Start-of-line' mode |
|:-|
| <img src="https://github.com/flitto/express-param/assets/67457712/a0c8742e-0c90-40cb-87de-9d3b14b2c623" width="900"> |

## 3) Text Region Recognition
- 기존에는 텍스트가 렌더링될 수 있는 영역을 bounding box가 차지하는 영역으로만 한정했습니다. 그러면 한국어를 영어로 번역할 때와 같이 원문보다 번역문이 긴 경우에는 필연적으로 원본보다 작은 font size로 텍스트를 렌더링할 수밖에 없습니다.
- text region recognition은 bounding box가 차지하는 영역을 벗어나 더욱 큰 font size를 가지고 텍스트를 렌더링할 수 있는지 계산하는 알고리즘입니다 [2].

| Original image | Human-annotated bounding boxes | Text-removed image |
|:-|:-|:-|
| <img src="https://github.com/KimRass/KimRass/assets/67457712/47052c9d-2e5d-47ef-949d-7786fee95709" width="500"> | <img src="https://github.com/KimRass/KimRass/assets/67457712/a84fae9d-57ee-44d6-b067-62a90dd0f6dd" width="500"> | <img src="https://github.com/KimRass/KimRass/assets/67457712/8f61041a-756a-43b3-b010-e7221c75ce8e" width="500"> |

### (1) Mask Generation
- 렌더링할 텍스트가 이미지 상의 주요 오브젝트와 충돌하지 않도록 제한하는 과정이 필요합니다. 이를 위해 3가지 마스크를 사용합니다.

| Image boundary mask | Edge mask [2] | Mask for remaining text | Intermediate mask |
|:-|:-|:-|:-|
| <img src="https://github.com/KimRass/KimRass/assets/67457712/79864008-da43-4ba3-b560-3c55968619a2" width="400"> | <img src="https://github.com/KimRass/KimRass/assets/67457712/8467efc8-cee9-4a2c-a7f8-c20cbd5e6ccc" width="400"> | <img src="https://github.com/KimRass/KimRass/assets/67457712/6485b16b-b954-45a1-931e-955132b201f8" width="400"> | <img src="https://github.com/KimRass/KimRass/assets/67457712/5f4286b7-bfbe-4318-848d-f7fe9113b861" width="400"> |
| font size를 점진적으로 확대해 나갈 때 텍스트가 이미지 바깥으로 나가지 않도록 하기 위해 사용됩니다. | 텍스트가 이미지에 존재하는 Edge를 넘어가지 않도록 제한하기 위해 사용됩니다. | 가격이나 음식점 이름과 같이 '메뉴 번역' 서비스의 정책상 제거하지 않은 텍스트와, 렌더링하고자 하는 텍스트가 서로 겹치지 않도록 하기 위해 사용됩니다. | 위 3개의 마스크는 렌더링할 텍스트에 무관하게 변하지 않으므로 빠른 계산을 위해 모두 합쳐 intermediate mask를 생성합니다. |

- 이 모든 마스크의 합과 렌더링할 타겟 텍스트가 충돌하지 않는 범위 내에서 조금씩 font size를 증가시킵니다. 이때 원본 font size보다 크게 증가시키지는 않습니다.
- 이때 4번마스크는 렌더링할 타겟 텍스트 자기 자신을 제외한 나머지 텍스트로 합니다. 이 4가지 마스크들의 합으로 새로운 마스크를 정의합니다.
- 추출된 font size의 절반에서 시작해 원본 font size의 0.6배 → 0.7배 → 0.8배 → 0.9배 → 1.0배 순으로 총 6개의 font size를 가지고 텍스트 렌더링을 시도합니다.
- 1번에서 정의한 마스크와 렌더링한 텍스트가 서로 충돌하는지 (겹침이 발생하는지) 확인합니다. 충돌이 발생하지 않으면 해당 font size를 해당 텍스트의 font size로 재정의합니다.
- 충돌 판별 로직:
    - 텍스트가 길 경우 줄바꿈의 수가 주어졌을 때 가능한 줄바꿈의 경우의 수는 매우 많습니다. 따라서 이 모든 경우를 하나씩 렌더링해보는 것은 매우 많은 계산량을 요구합니다.
    - 텍스트를 공백을 기준으로 서브워드 단위로 분리합니다. 분리된 서브워드들을 하나씩 붙여가면서 원래의 텍스트에 가깝게 조금씩 문자열을 만들어갑니다.
    - 완성되어가는 문자열을 가지고 렌더링을 시도합니다. 충돌이 발생하지 않으면 다음 서브워드를 그대로 이어붙여 다음 문자열을 만들고 충돌이 발생하면 줄바꿈을 중간에 삽입한 체로 다음 서브워드를 이어붙입니다. 충돌이 2회 연속 발생하면 해당 font size로의 확대에 실패한 것이므로 해당 bounding box의 최대 font size는 바로 직전 단계에서 텍스트 렌더링에 성공한 font size가 됩니다.

### (2) Examples
| Before and after |
|:-|
| <img src="https://github.com/KimRass/KimRass/assets/67457712/cb3ef1ba-aeb3-4f5a-a14c-7bf986cef64b" width="900"> |
