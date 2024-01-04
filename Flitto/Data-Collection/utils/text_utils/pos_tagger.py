from typing import List


class POSTagger(object):
    def __init__(
        self,
        logger=None,
        uses_spacy=False,
        uses_nltk=False,
        uses_flair=False,
        uses_khaiii=False,
        uses_mecab=False,
        uses_kiwi=False,
        typos="basic"
    ):
        self.logger = logger
        self.dic_kiwi_pos = {
            "N": ["NNG", "NNP", "NNB", "NR", "NP"],
            "V": ["VV", "VA", "VX", "VCP", "VCN"],
            "MA": ["MAG", "MAJ"],
            "J": ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"],
            "E": ["EP", "EF", "EC", "ETN", "ETM"],
            "XS": ["XSN", "XSV", "XSA"],
            "S": ["SF", "SP", "SS", "SE", "SO", "SW", "SL", "SH", "SN"],
            "W": ["W_URL", "W_EMAIL", "W_HASHTAG", "W_MENTION"]
        }

        if uses_spacy:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        elif uses_nltk:
            import nltk
            from nltk.tokenize import word_tokenize

            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
        elif uses_flair:
            from flair.data import Sentence
            from flair.models import SequenceTagger

            self.tagger = SequenceTagger.load("flair/pos-english-fast")
        elif uses_khaiii:
            from khaiii import KhaiiiApi

            self.api = KhaiiiApi()
        elif uses_mecab:
            from konlpy.tag import Mecab

            self.mcb = Mecab()
        elif uses_kiwi:
            from kiwipiepy import Kiwi

            self.kiwi = Kiwi(
                integrate_allomorph=True,
                model_type="sbg",
                typos=typos
            )

    def get_list_of_pos(self, sentence, method):
        if method == "spacy":
            return [token.tag_ for token in self.nlp(sentence)]
        elif method == "nltk":
            tokens = word_tokenize(sentence.lower())
            return [pos for _, pos in nltk.pos_tag(tokens)]
        elif method == "flair":
            sent = Sentence(sentence)
            self.tagger.predict(sent)
            return [token.get_label("pos").value for token in sent]
        elif method == "khaiii":
            return [morph.tag for token in self.api.analyze(sentence) for morph in token.morphs]
        elif method == "mecab":
            return [pos for _, pos in self.mcb.pos(sentence)]
        elif method == "kiwi":
            return [token.tag for token in self.kiwi.tokenize(sentence, normalize_coda=True)]
        else:
            return list()

    def get_string_of_pos(self, sentence, method):
        ls_pos = self.get_list_of_pos(sentence=sentence, method=method)
        return " ".join(ls_pos)

    def get_list_of_pos_with_word(self, sentence, method) -> List:
        if method == "kiwi":
            ls_word_pos = [
                f"{token.form}ᴥ{token.tag}"
                for token
                in self.kiwi.tokenize(sentence, normalize_coda=True)
            ]
            return ls_word_pos
        else:
            return list()

    def get_string_of_pos_with_word(self, sentence, method) -> str:
        if method == "kiwi":
            ls_word_pos = self.get_list_of_pos_with_word(
                sentence=sentence, method=method
            )
            str_word_pos = "|".join(ls_word_pos)
            str_word_pos = "|" + str_word_pos + "|"
            return str_word_pos
        else:
            return str()

    def get_sentence_separated_by_morphemes(self, sentence, method) -> str:
        if method == "kiwi":
            sentence_new = " ".join(
                [
                    f"|{token.form}|"
                    for token
                    in self.kiwi.tokenize(sentence, normalize_coda=True)
                ]
            )
            return sentence_new

    def get_sentence_separated_by_meaningful_morphemes(self, sentence, method) -> str:
        if method == "kiwi":
            sentence_new = " ".join(
                [
                    f"|{token.form}|"
                    for token
                    in self.kiwi.tokenize(sentence, normalize_coda=True)
                    if (
                        token.tag not in (
                            self.dic_kiwi_pos["V"]
                            + self.dic_kiwi_pos["MA"]
                            + self.dic_kiwi_pos["J"]
                            + self.dic_kiwi_pos["E"]
                            + ["XSV", "XSA", "W", "SF", "SP", "SS"]
                        )
                        and token.form not in ["·"]
                    )
                ]
            )
            return sentence_new

    def get_list_of_general_nouns_and_proper_nouns(self, sentence, method):
        if method == "kiwi":
            return [
                token.form
                for token
                in self.kiwi.tokenize(sentence, normalize_coda=True)
                if token.tag in ["NNG", "NNP"]
            ]
