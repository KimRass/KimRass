import re
from hanspell import spell_checker
from kiwipiepy import Kiwi
from collections import Counter

from utils.text_utils.text_normalizer import remove_tab_line_break_or_multiple_whitespaces

kiwi = Kiwi(
    integrate_allomorph=True,
    model_type="sbg",
    # typos=None
    typos="basic"
)


def detect_specific_words(sentence):
    pattern = "카톡|페메"
    return re.search(pattern, sentence)


def detect_specific_emoticons(sentence):
    pattern = ":D|T_T|T^T|OTL|oTL"
    return re.search(pattern, sentence)


def correct_wrong_characters(sentence):
    sentence = re.sub(pattern=r"떄", repl="때", string=sentence)
    sentence = re.sub(pattern=r"헀", repl="했", string=sentence)
    return sentence


def correct_only_wrong_spacings(sentence):
    sentence_no_spacing = remove_tab_line_break_or_multiple_whitespaces(sentence)

    try:
        sentence_spell_checked = spell_checker.check(sentence).checked
    except Exception:
        sentence_spell_checked = sentence

    sentence_spell_checked_no_spacing = remove_tab_line_break_or_multiple_whitespaces(
        sentence_spell_checked
    )

    if sentence_no_spacing != sentence_spell_checked_no_spacing:
        return sentence_spell_checked
    else:
        return sentence


def get_tags(sentence):
    ls_tag = [
        f"{token.form}ᴥ{token.tag}"
        for token
        in kiwi.tokenize(sentence, normalize_coda=True)
        if token.tag in ["NNG", "NNP", "VV", "VV-I", "VV-R", "VA-R", "VA-I", "XR", "SL", "SH", "SN"]
    ]
    ls_tag = [i for i in ls_tag if i not in ["하ᴥVV", "되ᴥVV"]]
    counter = Counter(ls_tag)
    return counter


def get_number_of_common_morphemes(sentence1, sentence2):
    counter1 = get_tags(sentence1)
    counter2 = get_tags(sentence2)
    intersection = counter1 & counter2
    return len(intersection)
