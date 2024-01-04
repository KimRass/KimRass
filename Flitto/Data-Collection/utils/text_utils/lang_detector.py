import langid
import fasttext
import google.cloud.translate_v2 as translate
from collections import Counter

fasttext.FastText.eprint = lambda x: None

unk_lang = "und"

pretrained_model = "/lid.176.ftz"
model = fasttext.load_model(pretrained_model)

translate_client = translate.Client()


def detect_lang_using_google(text):
    try:
        return translate_client.detect_language(text)["language"]
    except Exception as e:
        return unk_lang


def detect_lang_using_langid(text):
    try:
        return langid.classify(text)[0]
    except Exception as e:
        return unk_lang


def detect_lang_using_fasttext(text):
    try:
        return model.predict(text, k=1)[0][0].replace("__label__", "")
    except Exception as e:
        return unk_lang


def detect_langs(text):
    langs = [
        detect_lang_using_langid(text),
        detect_lang_using_fasttext(text),
        # detect_lang_using_google(text)
    ]
    
    langs = list(filter(lambda x: x != unk_lang, langs))
    
    if langs:
        langs_counter = Counter(langs)
        max_count = max(langs_counter.values())
        return list(filter(lambda x: langs_counter[x] == max_count, langs_counter))
    else:
        return list()


def are_lang_and_text_matched(lang, text):
    similar_lang_groups = [["id", "ms"], ["hi", "ne"], ["yue", "zh"]]

    detected_langs = detect_langs(text)

    for lang_group in similar_lang_groups:
        if (
            lang in lang_group
            and len(
                set(lang_group) & set(detected_langs)
            ) != 0
        ):
            return True, detected_langs

    if lang in detected_langs:
        return True, detected_langs
    elif not detected_langs:
        return False, detected_langs
    else:
        return False, [unk_lang]
