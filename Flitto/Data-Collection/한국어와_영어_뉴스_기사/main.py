PAT_EN = r"(VBD|VBP|VBZ|MD|HVS)"
PAT_KO1 = r"(EF|EC|ETN)($|( [A-Z]+)$)"
PAT_KO2 = r"SP[A-Z ]+SN SW[A-Z ]+NNG"
PAT_KO3 = r"SC[A-Z ]+SN SY[A-Z ]+NNG"

method_en1 = "spacy"
method_en2 = "nltk"
method_ko1 = "khaiii"
method_ko2 = "mecab"