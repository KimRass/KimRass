import html
import re
import jaconv
import string
from textacy import preprocessing


def remove_emojis(text):
    text = preprocessing.replace.emojis(text, "")
    return text


def remove_leading_or_trailing_whitespaces(text):
    text = text.strip()
    return text


def remove_tab_line_break_or_multiple_whitespaces(text):
    text = re.sub(pattern=r"\t|\n| +", repl="", string=text)
    return text


def replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace(text):
    text = re.sub(pattern=r"\t|\n| {2,}", repl=" ", string=text)
    return text


def standardize_punctuation_marks(text):
    dic = {
        r"[‘’ʼ´`ʻ]": "'",
        r"[“”ˮ″]": '"',
        r"[，、]": ","
    }
    for before, after in dic.items():
        text = re.sub(pattern=before, repl=after, string=text)
    return text


def correct_wrong_punctuation_marks(text):
    dic = {
        r"!\.|\.!|!,|,!": "!",
        r"\?\.|\.\?|\?,|,\?": "?",
        r"~\.|\.~|~,|,~": "~",
    }
    for before, after in dic.items():
        text = re.sub(pattern=before, repl=after, string=text)
    return text


def correct_repetitive_punctuation_marks(text):
    dic = {
        r"\~+": "~",
        r"\?+": "?",
        r"\!+": "!",
        r"\;+": ";",
        r"\,+": ",",
    }
    for before, after in dic.items():
        text = re.sub(pattern=before, repl=after, string=text)
    return text


def unescape_html_characters(text):
    return html.unescape(text)


def remove_invisible_unicode_characters(text):
    text = re.sub(
        pattern="\u00a0|\u00ad|\u034f|\u061c|\u17b4|\u17b5|\u180e|\u2000|\u2001|\u2002|\u2003|\u2004|\u2005|\u2006|\u2007|\u2008|\u2009|\u200a|\u200b|\u200c|\u200d|\u200e|\u200f|\u202f|\u205f|\u2060|\u2061|\u2062|\u2063|\u2064|\u206a|\u206b|\u206c|\u206d|\u206e|\u206f|\u3000|\u3164|\ufeff|\uffa0",
        repl="",
        string=text
    )
    return text


def remove_consecutive_single_quotes(text):
    text = re.sub(pattern=r"'{2,}", repl="'", string=text)
    return text


def full_width_to_half_width(text, lang="en"):
    lang = lang.lower()
    if lang not in ["ja", "zh", "zhcn", "zhtw", "zhsg", "zhhk"]:
        text = jaconv.z2h(text, kana=False, ascii=True, digit=True)
    return text


def remove_punctuation_marks(text):
    text = text.translate(text.maketrans("", "", string.punctuation))
    return text
