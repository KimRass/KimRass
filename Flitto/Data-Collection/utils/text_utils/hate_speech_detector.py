import re
from tqdm.auto import tqdm
from kiwipiepy import Kiwi
from typing import List


class HateSpeechDetector(object):
    def __init__(self):
        self.kiwi = Kiwi(
            integrate_allomorph=True,
            model_type="sbg",
            typos="basic",
        )
        self.hate_words_ko = {word.strip() for word in open("words_ko.txt", mode="r", encoding="utf-8")}
        self.hate_morphemes = {
            "자살NNG",
            "새끼NNG",
            "놈NNB",
            "짜증NNG",
            "개짱나NNG",
            "죽VV|고EC|싶VX",
            "죽이VV|고EC",
            "꺼지VV|어EC",
            "죽이VV|어EC",
            "신경질NNG",
            "미치VV",
            "놈NNB",
            "쓰레기NNG"
        }

    def convert_sentence_to_string_of_morphemes(self, sentence):
        return "|".join(
            [f"{token.form}{token.tag}" for token in self.kiwi.tokenize(sentence, normalize_coda=True)]
        )

    def detect_hate_speech_with_hate_morphemes(self, sentence):
        for word in self.hate_morphemes:
            if word in self.convert_sentence_to_string_of_morphemes(sentence):
                return True
        else:
            return False

    def detect_hate_speech_with_hate_words(self, sentence):
        for word in self.hate_words_ko:
            if word in sentence:
                return True
        else:
            return False

    def detect_hate_speech_with_all(self, sentence):
        if not self.detect_hate_speech_with_hate_morphemes(sentence):
            return self.detect_hate_speech_with_hate_words(sentence)
        else:
            return True
