import spacy
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences


class DataPreparation:
    def __init__(self) -> None:
        # for language processing
        self.nlp = spacy.load("en_core_web_md")

        # for cleaning the  reviews
        # words that can change the sentiment
        self.negative_words = [
            "n't",
            "not",
            "no",
            "never",
            "nothing",
            "nowhere",
            "noone",
            "none",
            "neither",
            "nor",
            "hardly",
            "scarcely",
            "barely",
            "without",
            "least",
            "few",
            "several",
            "less",
        ]
        self.positive_words = [
            "amazing",
            "awesome",
            "beautiful",
            "best",
            "brilliant",
            "celebrate",
            "charming",
            "cheerful",
            "congratulations",
            "delight",
            "delightful",
            "enjoy",
            "excellent",
            "exciting",
            "fabulous",
            "fantastic",
            "fun",
            "glamorous",
            "good",
            "gorgeous",
            "great",
            "happy",
            "hilarious",
            "impressive",
            "incredible",
            "inspiring",
            "jubilant",
            "joyful",
            "lovely",
            "marvelous",
            "nice",
            "perfect",
            "pleasurable",
            "positive",
            "promising",
            "remarkable",
            "rewarding",
            "satisfying",
            "splendid",
            "successful",
            "superb",
            "terrific",
            "thrilling",
            "triumphant",
            "vibrant",
            "wonderful",
        ]
        self.prepare_nlp_model()

    def tokenize_and_pad_items(self, item_seq=None, max_length=70):
        """tokenize the items, converts to the sequence and returns the sequence"""
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(item_seq)
        item_seq = tokenizer.texts_to_sequences(item_seq)
        item_seq = pad_sequences(item_seq, maxlen=max_length, truncating="pre")
        return item_seq

    # add arbritary number of words to stop words
    def add_to_stop_words(self, *args, nlp: "language_model"):
        for item in args:
            nlp.Defaults.stop_words.add(item)

    # remove arbritary number of item form the stop words
    def remove_from_stop_words(self, *args, nlp: "language_model"):
        for item in args:
            nlp.Defaults.stop_words.remove(item)

    # check if the words is in stop words lists:
    def check_word_in_stop_list(self, stop_list, word_list):
        for word in word_list:
            if word in stop_list:
                print(f"{word} is in stop list")
                # remove it from the stop list
                self.remove_from_stop_words(word, nlp=self.nlp)

    def prepare_nlp_model(self):
        # adding unecessay words to stopping words
        self.add_to_stop_words(
            ".",
            "-",
            "+",
            "\\",
            "/",
            "_",
            "'",
            '"',
            "?",
            ";",
            ":",
            "!",
            ",",
            " ",
            "I",
            "(",
            ")",
            "--",
            "[",
            "]",
            "the",
            "this",
            "however",
            "a",
            "he",
            "  ",
            "...",
            "we",
            nlp=self.nlp,
        )

        # checking removing the important negative words from stop list
        self.check_word_in_stop_list(
            word_list=self.negative_words, stop_list=self.nlp.Defaults.stop_words
        )
        self.check_word_in_stop_list(
            word_list=self.positive_words, stop_list=self.nlp.Defaults.stop_words
        )

    def clean_and_lemmitize(self, sentence):
        tokens = []
        # tokenize the data
        words_list = self.nlp(sentence)
        print(f"w-->{type(words_list)}")

        # clean and the data
        for words in words_list:
            print(words.text)

            if words.text not in self.nlp.Defaults.stop_words:
                t = words.lemma_

                if words.text == "n't":
                    t = "not"

                tokens.append(t.lower())

        print(f"t-->{tokens}")
        return tokens
