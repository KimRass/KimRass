import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import re
from itertools import product
from copy import deepcopy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", required=False)

    args = parser.parse_args()
    return args


def concatenate_dicts(dic_dir) -> None:
    dic_dir = Path(dic_dir)

    print("Concatenating all words...")

    dfs = list()
    for dic_path in tqdm(sorted((dic_dir / "words").glob("*.xls"))):
        if str(dic_path.name)[: 2] != "~$":
            df_dict = pd.read_excel(dic_path)
            dfs.append(df_dict)
    df_dic = pd.concat(dfs)
    
    print("Completed concatenating all words.")
    return df_dic


def get_variations_for_word(word, original="nan"):
    pattern = r"[-^ ]"
    substrs = re.split(pattern=pattern, string=word)
    substrs = list(map(lambda x: f"|{x}|", substrs))
    n = len(re.findall(pattern=pattern, string=word))

    vars = list()
    for spacing_case in product(["", " "], repeat=n):
        var =  "".join(
            [x for y in zip(substrs, spacing_case) for x in y] + [substrs[-1]]
        )
        var = var.replace("||", "").strip()
        vars.append(var)

    if isinstance(original, str) and original != "nan":
        original = original.replace("←", "")
        original = original.lower()
        match = re.search(pattern=f"[a-z ]+", string=original)
        if match:
            original_en = match.__getitem__(0)
            if original_en == original:
                vars.append("|" + "| |".join(original.split(" ")) + "|")
    return vars


def add_variations_to_dict(df_dict, dic_dir) -> None:
    dic_dir = Path(dic_dir)

    save_path = dic_dir / "opendict.pkl"
    print("Adding variations to each word...")

    rows = list()
    for row in tqdm(df_dict.values):
        row = list(row)
        word = row[0]
        vars = get_variations_for_word(word)
        for variation in vars:
            row_copied = deepcopy(row)
            row_copied.append(variation)
            row_copied.append(len(variation))

            rows.append(row_copied)
    df_dict = pd.DataFrame(
        rows, columns=df_dict.columns.tolist() + ["variation", "length_of_variation"]
    )
    df_dict.reset_index(drop=True, inplace=True)
    
    df_dict.to_pickle(save_path)
    print(f"Saved '{save_path.name}'.")


def main():
    args = get_args()

    df_dict = concatenate_dicts(args.dir)
    df_dict["standard"] = df_dict["어휘"].apply(lambda x: x.replace("^", " ").replace("-", ""))
    add_variations_to_dict(df_dict=df_dict, dic_dir=args.dir)
    

if __name__ == "__main__":
    main()
