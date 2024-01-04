import re

dic_lang_regex = {
    "ar": r"[\u0621-\u064A]+",
    "hi": r"[\u0900-\u097F]+",
    "ja": r"[ぁ-ゔァ-ヴー々〆〤ｧ-ﾝﾞﾟ]+",
    "km": r"[\u1780-\u17FF\u19E0-\u19FF]+",
    "ko": r"[ㄱ-ㅎㅏ-ㅣ가-힣]+",
    "ne": r"[\u0900-\u097F]+",
    "ru": r"[\u0410-\u044F\u0500-\u052F\u0400-\u04FF]+",
    "th": r"[\u0e01-\u0e5b]+",
    "tr": r"[a-zA-ZĞÜŞÖÇİğüşöçı]+",
    "vi": r"[a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+",
    "yue": r"[\u4e00-\u9fff]+",
    "zh": r"[\u4e00-\u9fff]+",
    "zhcn": r"[\u4e00-\u9fff]+",
    "zhtw": r"[\u4e00-\u9fff]+",
    "alphabet_except_en": r"[À-ÿ]+",
    "default": r"[a-zA-ZÀ-ÿ]+",
}

dic_lang_regex_unique = {
    "ar": r"[\u0621-\u064A]+",
    "hi": r"[\u0900-\u097F]+",
    "ja": r"[ぁ-ゔァ-ヴー々〆〤ｧ-ﾝﾞﾟ]+",
    "km": r"[\u1780-\u17FF\u19E0-\u19FF]+",
    "ko": r"[ㄱ-ㅎㅏ-ㅣ가-힣]+",
    "ru": r"[\u0410-\u044F\u0500-\u052F\u0400-\u04FF]+",
    "th": r"[\u0e01-\u0e5b]+",
    "vi": r"[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+"
}


def count_chars(text, lang="", lang_regex=None, uses_default_if_not_defined=True, unique=False):
    if unique:
        dic = dic_lang_regex_unique
    else:
        dic = dic_lang_regex

    all_chars_count = len(list(filter(lambda x: x.isalpha(), text)))

    if not lang_regex:
        if lang in dic:
            lang_regex = dic[lang]
        elif uses_default_if_not_defined:
            lang_regex = dic["default"]
        else:
            lang_regex = ""
    return sum(map(len, re.findall(lang_regex, text))), all_chars_count


def has_chars(text, lang="", lang_regex=None, uses_default_if_not_defined=True, unique=False):
    return count_chars(text, lang, lang_regex, uses_default_if_not_defined, unique=unique)[0] >= 1
