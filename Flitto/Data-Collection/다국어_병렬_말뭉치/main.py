import Levenshtein as lev

from utils.text_utils.text_normalizer import remove_invisible_unicode_characters


from utils.text_utils.text_normalizer import (
    remove_leading_or_trailing_whitespaces,
    replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace,
    standardize_punctuation_marks,
    correct_wrong_punctuation_marks,
    correct_repetitive_punctuation_marks,
    unescape_html_characters,
    remove_invisible_unicode_characters,
    remove_consecutive_single_quotes,
)


def refine_sentence(sentence):
    sentence = str(sentence)

    sentence = remove_leading_or_trailing_whitespaces(sentence)
    sentence = replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace(sentence)
    sentence = standardize_punctuation_marks(sentence)
    sentence = correct_wrong_punctuation_marks(sentence)
    sentence = correct_repetitive_punctuation_marks(sentence)
    sentence = unescape_html_characters(sentence)
    sentence = remove_invisible_unicode_characters(sentence)
    sentence = remove_consecutive_single_quotes(sentence)
    return sentence


def is_valid_quote_string(sentence):
    n_quote = sentence.count("'")
    n_double_quote = sentence.count('"')

    if (
        n_quote % 2 != 0 or
        n_double_quote % 2 != 0
    ):
        return False
    else:
        return True


def get_similarity(str1, str2):
    return round(
        1 - (2 * lev.distance(str1, str2) / (len(str1) + len(str2))), 3
    )
