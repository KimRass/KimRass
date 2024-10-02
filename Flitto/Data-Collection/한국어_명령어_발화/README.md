# 1. Overview
- "선풍기 켜줘", "온도 내려" 등의 명령어를 발화한 음성 데이터셋을 구축하는 프로젝트였습니다.

# 2. Requirements
- 확장자 'wav'
- 숨소리, 파열음 등의 잡음 제외
- 사람이 알아들을 수 있는 명확한 발음
- 음성 파일의 시작과 끝의 0.3초 이상의 무음 구간
- 1.5초 미만의 발화 구간
- 진폭 5,000 이상 20,000 미만
- 녹음되지 않은 고주파수 대역이 없을 것

# 3. Quality Assurance
- 다음 값들을 기준으로 음성 파일의 품질을 체크하여 품질이 떨어지는 음성 파일에 대해서만 한번 더 체크하도록 했습니다.
  - 발화 구간 전 무음 구간의 길이
  - 발화 구간의 전체 길이
  - 발화 구간 후 무음 구간의 길이
  - 오디오 파일 전체 길이에 대한 진폭 20,000 미만인 구간의 길이의 비율
  - 오디오 파일 전체 길이에 대한 진폭 5,000 이상인 구간의 길이의 비율
  - 미녹음된 주파수의 개수
  - 발화 구간의 개수
  - 특정한 알고리즘으로 측정한 Noise 수치
  - Noise가 심한 주파수
  - 기준을 충족하지 못한 조건의 개수
- Example
  | filename | sid | <div style="width: 80px">duration_of_silence_at_beginning</div> | <div style="width: 80px">total_duration_of_utterances</div> | <div style="width: 80px">duration_of_silence_at_end</div> | <div style="width: 80px">portion_under_maximum_amplitude</div> | <div style="width: 80px">portion_over_minimum_amplitude</div> | <div style="width: 80px">number_of_unrecorded_frequencies</div> | <div style="width: 80px">number_of_utterance_periods</div> | <div style="width: 80px">noise_value | <div style="width: 80px">number_of_frequencies_with_noise</div> | <div style="width: 80px">number_of_errors</div> |
  |-|-|-|-|-|-|-|-|-|-|-|-|
  | <div style="width:150px">female_122_010_female_(20)_c_index_speed_8401.wav</div> | 8401 | 0.47 | 0.72 | 0.37 | 1 | 0.252 | 2 | 1 | 38 | 1 | 0 |
  | <div style="width:150px">female_122_010_female_(20)_c_index_speed_8402.wav | 8402 | 0.39 | 0.91 | 0.3 | 1 | 0.231 | 1 | 1 | 49 | 1 | 0 |
  | <div style="width:150px">female_122_010_female_(20)_c_index_speed_8403.wav | 8403 | 0.42 | 0.56 | 0.4 | 1 | 0.362 | 3 | 1 | 80 | 1 | 0 |
  | <div style="width:150px">female_122_010_female_(20)_c_index_speed_8404.wav | 8404 | 0.41 | 0.83 | 0.46 | 1 | 0.307 | 4 | 1 | 73 | 1 | 0 |

# 4. udio Features
- ![features](https://i.imgur.com/tXk4ppa.png)
  - 위에서부터 순서대로 Waveform, Spectrogram, Mel-spectrogram입니다.
  - Waveform에서 상단에 각각 발화 구간 전 무음 구간의 길이, 발화 구간의 길이, 발화 구간 후 무음 기간의 길이를 표시했습니다.
  - Amplitude 5,000과 20,000에 대해서 수평선을 표시했습니다.
  - Spectrogram에서 y축의 파란색 표시는 Noise가 심한 주파수를, 빨간색 표시는 녹음되지 않은 주파수를 나타냅니다.
  - Spectrogram과 Mel-spectrogram에서 초록색 수직선은 발화 구간을 나타냅니다.
  - Mel-spectrogram에서 층을 이루는 것은 안정적인 발화를 나타냅니다.
