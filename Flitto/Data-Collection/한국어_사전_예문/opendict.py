import pandas as pd
import numpy as np
import json
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlencode


def look_up_in_opendict(
    word,
    type4="all",
    date_s=19000101,
    date_e=20990101
):
    url = "https://opendict.korean.go.kr/api/search"

    with open("/opendict_key.txt", mode="r") as f:
        key = f.read()
        f.close()
    query_params = "?" + urlencode(
        {
            quote_plus("key"): key,
            quote_plus("q"): word,
            quote_plus("req_type"): "json",
            quote_plus("num"): 100,
            quote_plus("part"): "word",
            # dict: 우리말샘순 popular: 많이 찾은 순 date: 새로 올린 순
            quote_plus("sort"): "date",
            quote_plus("advanced"): "y",
            # 1: 어휘(표제어) 2: 원어 3: 어원 4: 발음 5: 활용 6: 활용의 준말 7: 문형 8: 문법 9: 뜻풀이 10: 용례
            # 11: 용례 12: 대역어 13: 학명 14: 수어 정보 15: 규범 정보
            quote_plus("target"): 1,
            # `"all"`: 전체 general: 일상어 technical: 전문어 (다중 선택 가능)
            quote_plus("type4"): type4,
            quote_plus("date_s"): date_s,
            quote_plus("date_e"): date_e
        }
    )
    request = Request(url + query_params)
    request.get_method = lambda: "GET"
    response_body = json.load(urlopen(request))
    
    ls_sense = list()
    for idx in range(response_body["channel"]["num"]):
        dic = response_body["channel"]["item"][idx]["sense"][0]
        dic["type1"] = dic["type"]
        dic["type4"] = {"all": "전체", "general": "일상어", "technical": "전문어"}[type4]
        dic["word"] = word
        dic["origin"] = dic["origin"] if "origin" in dic else np.nan
        ls_sense.append(dic)
    cols = ["word", "sense_no", "definition", "type1", "type4", "pos", "origin", "target_code", "link"]
    if ls_sense:
        result = pd.DataFrame(ls_sense, columns=cols)
    else:
        dic = {
            "word": word,
            "sense_no": np.nan,
            "definition": np.nan,
            "type1": np.nan,
            "type4": np.nan,
            "pos": np.nan,
            "origin": np.nan,
            "target_code": np.nan,
            "link": np.nan
        }
        result = pd.DataFrame.from_records(dic, index=[0], columns=cols)

    result.sort_values(["sense_no"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
