import argparse
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from kiwipiepy import Kiwi
import copy
import re
from typing import List, Tuple

from utils.text_utils.text_normalizer import (
    standardize_punctuation_marks,
    replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace
)

tqdm.pandas()

kiwi = Kiwi(
    integrate_allomorph=True,
    model_type="sbg",
    typos="basic"
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel_path", type=str)
    parser.add_argument("--col", type=str)

    args = parser.parse_args()
    return args


def split_sentence_with_kiwi(sentence):
    return [
        sentence.text for sentence
        in kiwi.split_into_sents(sentence, normalize_coda=True, return_tokens=False)
    ]


def check_if_both_in_or_out(sentence_start, sentence_end, character_start, character_end):
    both_in = (sentence_start <= character_start and character_end < sentence_end)
    both_out = (character_end < sentence_start or sentence_end <= character_start)
    return (both_in or both_out) or (sentence_end - sentence_start < 2)


def chek_for_all_pairs(sentence_start, sentence_end, ls):
    for idx1, idx2 in ls:
        if not check_if_both_in_or_out(sentence_start, sentence_end, idx1, idx2):
            return False
    else:
        return True


def merge_sentences(sentence, ls_start_end, target="parenthesis") -> List[Tuple[str]]:
    if target == "parenthesis":
        pattern = r"""\([ㄱ-ㅎㅏ-ㅣ가-힣0-9 .!?,;'"]*\)"""
    elif target == "quote":
        pattern = r"""\'[ㄱ-ㅎㅏ-ㅣ가-힣0-9 .!?,;()"]*\'"""
    elif target == "double_quote":
        pattern = r"""\"[ㄱ-ㅎㅏ-ㅣ가-힣0-9 .!?,;()']*\""""
    ls_tup_target_start_target_end = [
        (i.start(0), i.end(0)) for i in re.finditer(pattern=pattern, string=sentence)
    ]
    
    ls_start_end_new = list()
    switch_prev = True
    for sentence_start, sentence_end in ls_start_end:
        switch = chek_for_all_pairs(sentence_start, sentence_end, ls_tup_target_start_target_end)

        if not switch:
            if switch_prev:
                ls_start_end_new.append((sentence_start, sentence_end))
            else:
                ls_start_end_new.append((ls_start_end_new.pop()[0], sentence_end))
        else:
            ls_start_end_new.append((sentence_start, sentence_end))
        switch_prev = switch
    return ls_start_end_new


def refine_sentence(sentence):
    sentence = standardize_punctuation_marks(sentence)
    sentence = replace_tab_line_break_or_multiple_whitespaces_with_single_whitespace(sentence)
    return sentence


def split_sentence(sentence):
    sentence = refine_sentence(sentence)

    ls_start_end = [
        (sentence.start, sentence.end) for sentence in kiwi.split_into_sents(
            sentence, normalize_coda=True, return_tokens=False
        )
    ]
    ls_start_end = merge_sentences(sentence, ls_start_end, target="parenthesis")
    ls_start_end = merge_sentences(sentence, ls_start_end, target="quote")
    ls_start_end = merge_sentences(sentence, ls_start_end, target="double_quote")
    
    sents = [
        sentence[sentence_start: sentence_end] for sentence_start, sentence_end in ls_start_end
    ]
    return sents


def remove_forefront_double_quote(sentence):
    sents = split_sentence(sentence)
    sents = [
        split_sentence(sentence[1:]) if sentence[0] == '"' else split_sentence(sentence)
        for sentence in sents
    ]
    sents = sum(sents, [])
    return sents


def add_splitted_sentences_column(df, col) -> pd.DataFrame:
    col_idx = df.columns.tolist().index(col)
    
    rows = list()
    for row in tqdm(df.values):
        row = list(row)
        sentence = row[col_idx]
        sents = split_sentence(sentence)
        sents = remove_forefront_double_quote(sentence)
        for sentence2 in sents:
            # print(sentence2, end="\n\n")
            row_copied = copy.deepcopy(row)
            row_copied.append(sentence2)
            rows.append(row_copied)
    return pd.DataFrame(rows, columns=df.columns.tolist() + [f"{col}_splitted"])


def main():
    args = get_args()
    
    df = pd.read_excel(args.excel_path)

    df = add_splitted_sentences_column(df=df, col=args.col)

    df.to_excel(Path(args.excel_path).parent / f"{Path(args.excel_path).stem}_sentence_splitted.xlsx", index=False)


if __name__ == "__main__":
    main()
