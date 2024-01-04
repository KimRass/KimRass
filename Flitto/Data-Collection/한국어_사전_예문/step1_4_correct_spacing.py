from tqdm.auto import tqdm
import copy
from pathlib import Path
import pandas as pd

from b2b_projects.NIKL_2022_01.utilities import get_arguments
from libs.utils.logger import Logger

tqdm.pandas()


def prepare_words(dir):
    df_words = pd.read_pickle(dir / "words_concatenated_variation.pkl")
    df_words = df_words[df_words["어휘"].str.contains(r"[-^ ]")]
    df_words.drop_duplicates(subset=["variation"], keep="first", inplace=True)
    return df_words


def concat_sentence_variation(dir, target):
    logger.info(f"Concatenating '{target}_sentence_variation_*.xlsx'...")

    dfs = list()
    for sentence_variation_xlsx_path in tqdm(list(dir.glob(f"{target}/*.xlsx"))):
        if (
            f"{target}_sentence_variation" in str(sentence_variation_xlsx_path) and
            sentence_variation_xlsx_path.stem[: 2] != "~$"
        ):
            df_sentence_variation = pd.read_excel(sentence_variation_xlsx_path)
            dfs.append(df_sentence_variation)
    df_concated = pd.concat(dfs)
    df_concated.sort_values(["translation_id"], inplace=True)

    logger.info(f"Completed concatenating all of '{target}_sentence_variation_*.xlsx'.")
    return df_concated


def remove_substring_variations(df):
    ls_row = list()
    for (tid, sentence), group in df.groupby(["translation_id", "원문"]):
        for variation1 in group["variation"].values:
            for variation2 in group["variation"].values:
                if (
                    variation1 != variation2 and
                    variation1 in variation2
                ):
                    break
            else:
                ls_row.append((tid, sentence, variation1))
    return pd.DataFrame(ls_row, columns=df.columns.tolist())


def merge_sentence_variation_and_words(df_sentence_variation, df_words):
    df_merged = pd.merge(
        df_sentence_variation, df_words[["variation", "standard"]],
        on="variation"
    )
    df_merged.sort_values(["translation_id", "variation"], inplace=True)
    df_merged = df_merged[
        df_merged["variation_replaced"] != df_merged["standard"]
    ]
    df_merged.groupby(["translation_id"])["variation_replaced"].apply(list)

    df_merged = pd.merge(
        df_merged[["translation_id", "원문", "standard"]],
        df_merged.groupby(["translation_id"])["variation_replaced"].apply(list),
        on="translation_id"
    )
    df_merged = pd.merge(
        df_merged[["translation_id", "원문", "variation_replaced"]],
        df_merged.groupby(["translation_id"])["standard"].apply(list),
        on="translation_id"
    )
    df_merged.sort_values(["translation_id"], inplace=True)
    return df_merged


def correct_spacing(dir, target):
    dir = Path(dir)

    df_words = prepare_words(dir)

    df_sentence_variation = concat_sentence_variation(dir, target)
    df_sentence_variation = remove_substring_variations(df_sentence_variation)
    df_sentence_variation["variation_replaced"] = df_sentence_variation["variation"].apply(
        lambda x: x.replace("|", "")
    )

    df_merged = merge_sentence_variation_and_words(df_sentence_variation, df_words)
    df_merged.drop_duplicates(["translation_id"], inplace=True)

    ls_row = list()
    for tid, sentence, ls_variation_replaced, ls_standard in df_merged.values:
        sentence_ori = copy.deepcopy(sentence)

        for before, after in zip(ls_variation_replaced, ls_standard):
            sentence = sentence.replace(before, after)
        ls_row.append((tid, sentence_ori, sentence))
    df_replaced = pd.DataFrame(ls_row, columns=["translation_id", "before", "after"])
    df_replaced["are_same"] = df_replaced["before"].eq(df_replaced["after"])
    df_replaced.sort_values(["translation_id"], inplace=True)
    return df_replaced


def add_redundant_sentences(dir, target, df_replaced):
    dir = Path(dir)

    df_replaced.rename({"translation_id": "translation id"}, axis=1, inplace=True)

    df_sentences = pd.read_excel(dir / target / f"{target}.xlsx")
    df_sentences["before"] = df_sentences["원문"]
    df_sentences["after"] = df_sentences["원문"]
    df_sentences["are_same"] = True
    df_sentences = df_sentences[df_replaced.columns.tolist()]

    df = pd.concat([df_replaced, df_sentences])
    df.drop_duplicates(subset=["translation id"], keep="first", inplace=True)
    df.sort_values(["translation id"], inplace=True)

    replaced_xlsx_path = dir / target / f"{target}_replaced.xlsx"
    df.to_excel(replaced_xlsx_path, index=False)
    logger.info(f"Saved '{replaced_xlsx_path.name}'.")


def main():
    args = get_arguments()

    global logger
    logger = Logger(args.dir).get_logger()

    df_replaced = correct_spacing(dir=args.dir, target=args.target)
    add_redundant_sentences(
        dir=args.dir, target=args.target, df_replaced=df_replaced
    )


if __name__ == "__main__":
    main()
