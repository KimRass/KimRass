from kiwipiepy import Kiwi


class FormalInformalConverter(object):
    def __init__(self):
        self.kiwi = Kiwi(
            integrate_allomorph=True,
            model_type="sbg",
            typos="basic"
        )
        self.dic_informal = {
            "|안녕ᴥNNG|하ᴥXSA|시ᴥEP|어요ᴥEF|": "|안녕ᴥNNG|",
            "|인사ᴥNNG|드리ᴥVV|ᆸ니다ᴥEF|": "|인사ᴥNNG|하ᴥXSV|어ᴥEF|",
            "|드리ᴥVV|ᆯᴥETM|말씀ᴥNNG|": "|하ᴥVV|어ᴥEC|주ᴥVX|ᆯᴥETM|말ᴥNNG|",
            "|부탁ᴥNNG|드리ᴥVV|ᆸ니다ᴥEF|": "|부탁ᴥNNG|하ᴥXSV|어ᴥEF|",
            "|사오ᴥEP|ᆸ니다ᴥEF|": "|어ᴥEF|",
            "|오ᴥEP|ᆸ니다ᴥEF|": "|어ᴥEF|",
            "|주ᴥVX|시ᴥEP|어요ᴥEF|": "|주ᴥVX|어ᴥEF|",
            "|어떠ᴥXR|하ᴥXSA|었ᴥEP|니ᴥEF|": "|어떻ᴥVA|었ᴥEP|니ᴥEC|",
            "|에요ᴥEF|": "|야ᴥEF|",
            "|어요ᴥEF|": "|아ᴥEF|",
            "|죄송하ᴥVA|": "|미안하ᴥVA|",
            "|습니까ᴥEF|": "|니ᴥEC|",
            "|습니다ᴥEF|": "|어ᴥEC|",
            "|죠ᴥEF|": "|지ᴥEF|",
            "|오시ᴥVV|": "|오ᴥVV|",
            "|예요ᴥEF|": "|야ᴥEF|",
            "|까요ᴥEF|": "|까ᴥEF|",
            "|ᆯ까요ᴥEF|": "|ᆯ까ᴥEF|",
            "|었ᴥEP|야ᴥEF|": "|었ᴥEP|어ᴥEF|",
            "|ᆸ니까ᴥEF|": "|야ᴥEF|",
            "|ᆸ니다ᴥEF|": "|어ᴥEF|",
            "|ᆫ가ᴥEF|요ᴥJX|": "|야ᴥEF|",
            "|ᆫ가요ᴥEF|": "|야ᴥEF|",
            "|네요ᴥEF|": "|네ᴥEF|",
            "|나요ᴥEF|": "|니ᴥEC|",
            "|드리ᴥVV|": "|주ᴥVV|",
            "|어ᴥEC|드리ᴥVX|": "|",
            "|시ᴥEP|ᆸ쇼ᴥEF|": "|아ᴥEF|",
            "|라뇨ᴥEF|": "|라니ᴥEF|",
            "|을까요ᴥEF|": "|을까ᴥEF|",
            "|ᆯ게요ᴥEF|": "|ᆯ게ᴥEF|",
            "|라고요ᴥEF|": "|라고ᴥEF|",
            "|으시ᴥEP|": "|",
            "|시ᴥEP|": "|",
            "|요ᴥJX|": "|",
            "|죄송ᴥXR|": "|미안ᴥNNG|",
            "|저ᴥNP|": "|나ᴥNP|",
            "|저ᴥIC|": "|나ᴥNP|",
            "|말씀ᴥNNG|": "|말ᴥNNG|"
        }
        self.dic_correction = {
            "|좋ᴥVA|": "|좋ᴥW_MENTION|",
            "|것ᴥNNB|이ᴥVCP|": "|것이ᴥW_MENTION|",
            "|이것ᴥNP|이ᴥVCP|": "|이것이ᴥW_MENTION|",
            "|그것ᴥNP|이ᴥVCP|": "|그것이ᴥW_MENTION|",
            "|저것ᴥNP|이ᴥVCP|": "|저것이ᴥW_MENTION|",
            "|거ᴥNNB|ᆫᴥJX|": "|건ᴥW_MENTION|",
            "|이르ᴥVV|어ᴥEC|": "|일러ᴥW_MENTION|",
            "|이렇ᴥVA|ᆫᴥETM|": "|이런ᴥMM|",
            "|않ᴥVX|아ᴥEF|": "|않아ᴥW_MENTION|",
            "|되ᴥVV|야ᴥEF|": "|돼ᴥW_MENTION|",
            "|어떻ᴥVA|야ᴥEF|": "|어때?ᴥW_MENTION|",
            "|있ᴥVX|니ᴥEF|": "|있니ᴥW_MENTION|",
            "|이ᴥVCP|어ᴥEF|": "|야ᴥEF|",
            "|말ᴥVX|어ᴥEC|": "|말아ᴥW_MENTION|",
            "|그러ᴥVV|아ᴥEF|": "|그래ᴥW_MENTION|",
            "|꺼ᴥNNB|": "|거ᴥNNB|",
            "|어떻ᴥVA|ᆯ까ᴥEC|": "|어떨까ᴥW_MENTION|"
        }

    def convert_sentence_to_string_of_morphemes(self, sentence):
        str_morphemes = "|".join(
            [f"{token.form}ᴥ{token.tag}" for token in self.kiwi.tokenize(sentence, normalize_coda=True)]
        )
        str_morphemes = "|" + str_morphemes + "|"
        return str_morphemes

    def convert_formal_to_informal(self, sentence):
        str_morphemes = self.convert_sentence_to_string_of_morphemes(sentence)

        needs_converting = False
        for before, after in self.dic_informal.items():
            if before in str_morphemes:
                str_morphemes = str_morphemes.replace(before, after)
                needs_converting = True
        if not needs_converting:
            return sentence
        else:
            for before, after in self.dic_correction.items():
                if before in str_morphemes:
                    str_morphemes = str_morphemes.replace(before, after)
            str_morphemes = str_morphemes[1:-1]
            sentence = self.kiwi.join(
                [tuple(str_morpheme.split("ᴥ")) for str_morpheme in str_morphemes.split("|")]
            )
            sentence = sentence.replace("#챗봇이름 #", "#챗봇이름#")
            return sentence
