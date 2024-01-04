import argparse
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

tqdm.pandas()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", required=False)

    args = parser.parse_args()
    return args


def create_df_from_list_of_rows(list_of_rows) -> pd.DataFrame:
    df_sentence_variation = pd.DataFrame(
        list_of_rows, columns=["translation_id", "원문", "variation"]
    )
    df_sentence_variation.sort_values(["translation_id", "variation"], inplace=True)
    return df_sentence_variation


def get_index_created_at_last(dir, target):
    idx_row_to_max = -1
    for sentence_variation_xlsx_path in dir.glob(f"{target}/*.xlsx"):
        stem = sentence_variation_xlsx_path.stem
        if (
            stem.rsplit("_", 1)[0] == f"{target}_sentence_variation" and
            stem[: 2] != "~$"
        ):
            idx_row_to = int(stem.split("-")[-1])
            if idx_row_to > idx_row_to_max:
                idx_row_to_max = idx_row_to
    return idx_row_to_max


def get_df_words(dir) -> pd.DataFrame:
    df_words = pd.read_pickle(dir / f"words_concatenated_variation.pkl")

    df_words[["어휘", "품사"]] = df_words[["어휘", "품사"]].astype("category")
    df_words["전문 분야"] = df_words["전문 분야"].astype("str")
    df_words = df_words[
        df_words["품사"].isin(
            ["품사 없음", "명사", "관·명", "감·명", "명·부", "수·관·명"]
        )
    ]
    df_words = df_words[
        ["어휘", "원어·어종", "원어", "의미 번호", "뜻풀이", "전문 분야", "대역어", "규범 정보", "variation", "length_of_variation"]
    ]
    return df_words


def get_number_of_digits(df_sentences) -> int:
    n_sentence = len(df_sentences)
    n_digit = len(str(n_sentence))
    return n_digit


def create_sentence_variation_xlsx(dir, target, chunk_size=30_000) -> None:
    print(f"Creating '{target}_sentence_variation_*.xlsx'...")

    dir = Path(dir)

    df_words = get_df_words(dir)

    df_sentences = pd.read_excel(
        dir / target / f"{target}_preprocessed.xlsx",
        usecols=["translation id", "원문", "원문_converted", "원문_joined"]
    )
    df_sentences[["원문", "원문_converted", "원문_joined"]] = df_sentences[["원문", "원문_converted", "원문_joined"]].astype("str")

    n_digit = get_number_of_digits(df_sentences)

    last_idx = get_index_created_at_last(dir=dir, target=target)
    idx_row_prev = last_idx + 1

    ls_row = list()
    for idx_row, (tid, sentence, sentence_converted, sentence_joined) in tqdm(
        list(
            enumerate(
                df_sentences.iloc[last_idx + 1:].values,
                start=last_idx
            )
        )
    ):
        if len(ls_row) >= chunk_size:
            df_sentence_variation = create_df_from_list_of_rows(ls_row)

            sentence_variation_xlsx_path = dir / target / f"{target}_sentence_variation_{str(idx_row_prev).zfill(n_digit)}-{str(idx_row).zfill(n_digit)}.xlsx"
            df_sentence_variation.to_excel(sentence_variation_xlsx_path, index=False)
            print(f"Saved {len(df_sentence_variation)} rows to '{sentence_variation_xlsx_path.name}'.")

            ls_row = list()
            idx_row_prev = idx_row + 1

        for variation in df_words["variation"].unique():
            if (
                variation in sentence_converted and
                variation in sentence_joined
            ):
                ls_row.append([tid, sentence, variation])
    print(f"Completed creating all of '{target}_sentence_variation_*.xlsx'")


def merge_words_and_sentence_variation(df_words, df_sentence_variation) -> pd.DataFrame:
    df_step_b = pd.merge(df_words, df_sentence_variation, on="variation")
    df_step_b["구분"] = df_step_b["전문 분야"].apply(
        lambda x: "일상어" if x == "nan" else "전문어"
    )
    df_step_b = df_step_b[
        ["translation_id", "원문", "어휘", "원어·어종", "의미 번호", "뜻풀이", "전문 분야", "구분", "대역어", "규범 정보", "variation", "length_of_variation"]
    ]
    df_step_b.sort_values(["translation_id", "어휘", "의미 번호"], inplace=True)
    return df_step_b


def create_step_b_xlsx(dir, target) -> None:
    print(f"Creating '{target}_step_b_*.xlsx'...")

    dir = Path(dir)

    df_words = get_df_words(dir)

    for sentence_variation_xlsx_path in tqdm(list(dir.glob(f"{target}/*.xlsx"))):
        if (
            f"{target}_sentence_variation" in str(sentence_variation_xlsx_path) and
            sentence_variation_xlsx_path.stem[: 2] != "~$"
        ):
            df_sentence_variation = pd.read_excel(sentence_variation_xlsx_path)

            df_step_b = merge_words_and_sentence_variation(df_words, df_sentence_variation)

            step_b_xlsx_path = dir / target / f"{target}_step_b_{sentence_variation_xlsx_path.stem.rsplit('_', 1)[1]}.xlsx"
            df_step_b.to_excel(step_b_xlsx_path, index=False)
    print(f"Completed creating all of '{target}_step_b_*.xlsx'")


def main() -> None:
    args = get_args()

    create_sentence_variation_xlsx(dir=args.dir, target=args.target, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
