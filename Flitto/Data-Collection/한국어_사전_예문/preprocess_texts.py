import argparse
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

from utils.text_utils.pos_tagger import POSTagger

tqdm.pandas()

pos_tagger = POSTagger(uses_kiwi=True, typos="basic")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", required=True)
    parser.add_argument("--col", required=True)

    args = parser.parse_args()
    return args


def add_joined_sentence_column(df, col):
    print("Adding joined sentences column...")

    df[f"""{col}_joined"""] = df[col].progress_apply(
        lambda x: pos_tagger.get_sentence_separated_by_morphemes(
            sentence=x.lower(), method="kiwi"
        )
    )

    print("Completed adding joined sentences column.")


def add_converted_sentence_column(df, col):
    print("Adding converted sentences column...")

    df[f"""{col}_converted"""] = df[col].progress_apply(
        lambda x: pos_tagger.get_sentence_separated_by_meaningful_morphemes(
            sentence=x.lower(), method="kiwi"
        )
    )

    print("Completed adding converted sentences column.")


def prepare_sentences(target, col):
    df = pd.read_excel(target)
    add_converted_sentence_column(df=df, col=col)
    add_joined_sentence_column(df=df, col=col)

    save_path = Path(target).parent/f"""{Path(target).stem}_preprocessed.xlsx"""
    df.to_excel(save_path, index=False)
    print(f"Saved '{save_path.name}'.")


def main():
    args = get_args()

    prepare_sentences(target=args.target, col=args.col)


if __name__ == "__main__":
    main()
