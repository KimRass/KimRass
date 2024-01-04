from tqdm.auto import tqdm
import re
from textacy.preprocessing.resources import RE_URL, RE_SHORT_URL

from utils.text_utils.lang_detector import are_lang_and_text_matched
from utils.text_utils.lang_regex import has_chars
from utils.text_utils.text_normalizer import (
    remove_punctuation_marks,
    replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace
)

tqdm.pandas()

language_with_unique_characters = [
    "ar", "hi", "ja", "km", "ko", "ne", "ru", "th", "tr", "vi", "yue", "zh", "zhcn", "zhtw", "kh"
]
language_without_unique_characters = ["uz", "en", "tl", "id"]

language_with_spacing = ["ko", "en", "uz", "tl", "th", "hi", "vi", "id"]
language_without_spacing = [
    "ja", "zh", "zhcn", "zhtw", "zhsg", "zhhk", "yue", "th", "hi", "ne", "km", "th"
]

pattern_email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
pattern_mobile_ko = r"0[0-1][0-9]-[0-9]{3,4}-[0-9]{4}"
pattern_tel = r"0[02-8][0-9]{0,1}-[0-9]{3,4}-[0-9]{4}"
pattern_url=f"{RE_URL.pattern}|{RE_SHORT_URL.pattern}"


def is_not_duplicated(df, lang1, lang2) -> None:
    for lang in [lang1, lang2]:
        df["temp"] = df[lang]

        df["temp"] = df["temp"].astype("str")
        df["temp"] = df["temp"].fillna("")
        df["temp"] = df["temp"].apply(lambda x: x.replace(" ", ""))
        df["temp"] = df["temp"].str.lower()
        df["temp"] = df["temp"].map(
            replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace
        )
        df["temp"] = df["temp"].map(remove_punctuation_marks)

        df[f"{lang}_is_not_duplicated"] = ~df[lang].duplicated(keep=False)
        
    df.drop(["temp"], axis=1, inplace=True)


def matches_with_language(df, lang1, lang2) -> None:
    for lang in [lang1, lang2]:
        df[f"{lang}_matches_with_{lang}"] = df[lang].apply(
            lambda x: are_lang_and_text_matched(lang=lang, text=x)[0]
        )


def get_number_of_characters_or_words(df, lang1, lang2) -> None:
    for lang in [lang1, lang2]:
        if lang in language_with_spacing:
            df[f"number_of_{lang}_characters"] = df[lang].map(len)
        elif lang in language_without_spacing:
            df[f"number_of_{lang}_words"] = df[lang].apply(
                lambda x: len(x.split())
            )
        else:
            print(f"No idea if '{lang}' has spacing!")


def has_unique_character_of_itself(df, lang1, lang2) -> None:
    for lang in [lang1, lang2]:
        if lang in language_with_unique_characters:
            df[f"{lang}_has_unique_{lang}_character"] = df[lang].apply(
                lambda x: has_chars(
                    text=x, lang=lang, uses_default_if_not_defined=True, unique=True
                )
            )
        elif lang not in language_without_unique_characters:
            print(f"No idea if '{lang}' has unique characters!")


def not_has_unique_character_of_counterpart(df, lang1, lang2) -> None:
    if lang1 in language_with_unique_characters:
            df[f"{lang2}_not_has_unique_{lang1}_character"] = df[lang2].apply(
                lambda x: not has_chars(
                    text=x, lang=lang1, uses_default_if_not_defined=True, unique=True
                )
            )
    elif lang1 not in language_without_unique_characters:
            print(f"No idea if '{lang1}' has unique characters!")

    if lang2 in language_with_unique_characters:
            df[f"{lang1}_not_has_unique_{lang2}_character"] = df[lang1].apply(
                lambda x: not has_chars(
                    text=x, lang=lang2, uses_default_if_not_defined=True, unique=True
                )
            )
    elif lang2 not in language_without_unique_characters:
            print(f"No idea if '{lang2}' has unique characters!")


def not_has_hangul(df, lang1, lang2) -> None:
    if "ko" not in [lang1, lang2]:
        for lang in [lang1, lang2]:
            df[f"{lang}_not_has_hangul"] = df[lang].apply(
                lambda x: not has_chars(
                    text=x, lang="ko", uses_default_if_not_defined=True, unique=True
                )
            )


def not_has_pii(df, lang1, lang2) -> None:
    pattern = f"{pattern_email}|{pattern_mobile_ko}|{pattern_tel}|{pattern_url}"

    for lang in [lang1, lang2]:
        df[f"{lang}_not_has_pii"] = df[lang].apply(
            lambda x: False if re.search(pattern=pattern, string=x) else True
        )


def are_not_identical(df, lang1, lang2) -> None:
    df[f"{lang1}_and_{lang2}_are_not_identical"] = df.apply(
        lambda x: x[lang1] != x[lang2],
        axis=1
    )


def is_ratio_of_length_appropriate(df, lang1, lang2) -> None:
    df[f"length_of_{lang1}_over_lenght_of_{lang2}"] = df.apply(
        lambda x: round(len(x[lang1]) / len(x[lang2]), 2),
        axis=1
    )


def ends_with_appropriate_character(lang2, text1, text2, spoken_or_written="written"):
    if lang2 in ["km"]:
        chars = ["។", "?", "!", "'", '"']
    elif lang2 in ["hi"]:
        chars = ["।", "?", "!", "'", '"']
    elif lang2 in ["th"]:
        if text1[-1] == ".":
            return True
        else:
           chars = [text1[-1]]
        return True
    else:
        chars = [".", "?", "!", "'", '"']
        if spoken_or_written == "spoken":
            chars += ["~", "…"]

    if text2 and text2[-1] in chars:
        return True
    else:
        return False
