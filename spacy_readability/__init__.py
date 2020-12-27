# -*- coding: utf-8 -*-

"""Top-level package for spacy_readability."""

__author__ = """Michael Holtzscher"""
__email__ = "mholtz@protonmail.com"
__version__ = "1.4.2"

from math import sqrt
import numpy as np
from numpy.random import choice as random_sample
from spacy.tokens import Token
from spacy.tokens import Doc
from spacy_syllables import SpacySyllables


from .words import DALE_CHALL_WORDS
from .measure_parameters import MEASURE_PARAMETERS


class Readability(SpacySyllables):
    """spaCy v2.0 pipeline component for calculating readability scores of of text.
    Provides scores for Flesh-Kincaid grade level, Flesh-Kincaid reading ease, and Dale-Chall.
    USAGE:
        >>> import spacy
        >>> from spacy_readability import Readability
        >>> nlp = spacy.load('en')
        >>> read = Readability()
        >>> nlp.add_pipe(read, last=True)
        >>> doc = nlp("I am some really difficult text. I use obnoxiously large words.")
        >>> print(doc._.flesch_kincaid_grade_level)
        >>> print(doc._.flesch_kincaid_reading_ease)
        >>> print(doc._.dale_chall)
        >>> print(doc._.smog)
        >>> print(doc._.coleman_liau_index)
        >>> print(doc._.automated_readability_index)
        >>> print(doc._.forcast)
    """

    name = "readability"

    def __init__(self, nlp, lang=None, measures=None):
        """Initialise the pipeline component.
        """
        super(Readability, self).__init__(nlp, lang=lang)
        lang = lang or nlp.lang
        # take only supported measures
        if measures:
            self.measures = {metric: MEASURE_PARAMETERS[lang][metric] for metric in set(MEASURE_PARAMETERS[lang].keys()) & set(measures)}
        else:
            self.measures = MEASURE_PARAMETERS[lang]

        for metric in ["total_sentences", "total_words", "total_syllables", "total_letters"]:
            Doc.set_extension(metric, default=None, force=True)

        Token.set_extension("letters_count", default=None, force=True)

        for metric in self.measures.keys():
            if not Doc.has_extension(metric):
                Doc.set_extension(metric, getter=getattr(self, metric))


    def __call__(self, doc):
        total_sentences = 0
        total_words = 0
        total_syllables = 0
        total_letters = 0
        for sent in doc.sents:
            total_sentences += 1
            for token in sent:
                if not token.is_punct and not token.is_digit:
                    total_words += 1
                    token._.set("letters_count", len(token.text))
                    total_letters += len(token.text)
                    syllables = self.syllables(token.text)
                    if syllables:
                        token._.set("syllables", syllables)
                        token._.set("syllables_count", len(syllables))
                        total_syllables += len(syllables)
        doc._.set("total_sentences", total_sentences)
        doc._.set("total_words", total_words)
        doc._.set("total_syllables", total_syllables)
        doc._.set("total_letters", total_letters)
        return doc

    def flesch_kincaid_grade_level(self, doc: Doc):
        """Returns the Flesch-Kincaid grade for the document.
        """
        params = self.measures["flesch_kincaid_grade_level"]
        num_sentences = doc._.total_sentences
        num_words = doc._.total_words
        num_syllables = doc._.total_syllables
        if num_sentences == 0 or num_words == 0 or num_syllables == 0:
            return 0
        return (
            (params["asw"] * num_syllables / num_words)
            + (params["asl"] * num_words / num_sentences)
            - params["base"]
        )

    def flesch_kincaid_reading_ease(self, doc: Doc):
        """Returns the Flesch-Kincaid Reading Ease score for the document.
        """
        params = self.measures["flesch_kincaid_reading_ease"]
        num_sentences = doc._.total_sentences
        num_words = doc._.total_words
        num_syllables = doc._.total_syllables
        if num_sentences == 0 or num_words == 0 or num_syllables == 0:
            return 0
        words_per_sent = num_words / num_sentences
        syllables_per_word = num_syllables / num_words
        return params["base"] - (params["asl"] * words_per_sent) - (params["asw"] * syllables_per_word)

    def dale_chall(self, doc: Doc):
        """Returns the Dale-Chall score for the document.
        """
        params = self.measures["dale_chall"]
        num_sentences = doc._.total_sentences
        num_words = doc._.total_words
        if num_sentences == 0 or num_words == 0:
            return 0
        diff_words_count = sum(1 for word in doc if (
                    not word.is_punct
                    and not word.is_digit
                    and word.text.lower() not in DALE_CHALL_WORDS
                    and word.lemma_.lower() not in DALE_CHALL_WORDS
                )
        )
        percent_difficult_words = 100 * diff_words_count / num_words
        average_sentence_length = num_words / num_sentences
        grade = params["pdw"] * percent_difficult_words + params["asl"] * average_sentence_length

        # if percent difficult words is about 5% then adjust score
        if percent_difficult_words > 5:
            grade += 3.6365
        return grade

    def smog(self, doc: Doc):
        """Returns the SMOG score for the document. If there are less than 30 sentences then
        it returns 0 because he formula significantly loses accuracy on small corpora.
        """
        params = self.measures["smog"]
        num_sentences = doc._.total_sentences
        num_words = doc._.total_words
        if num_sentences < 30 or num_words == 0:
            return 0
        num_poly = _get_num_syllables(doc, min_syllables=3)
        return params["mult"] * sqrt(num_poly * 30 / num_sentences) + params["base"]

    def coleman_liau_index(self, doc: Doc):
        """Returns the Coleman-Liau index for the document."""
        params = self.measures["coleman_liau_index"]
        num_words = doc._.total_words
        if num_words <= 0:
            return 0

        num_sentences = doc._.total_sentences
        letter_count = doc._.total_letters
        if letter_count <= 0:
            return 0
        letters_to_words = letter_count / num_words * 100
        sent_to_words = num_sentences / num_words * 100
        return params["l"] * letters_to_words - params["s"] * sent_to_words - params["base"]

    def automated_readability_index(self, doc: Doc):
        """Returns the Automated Readability Index for the document.
        """
        params = self.measures["automated_readability_index"]
        num_sentences = doc._.total_sentences
        num_words = doc._.total_words
        if num_words <= 0:
            return 0

        letter_count = doc._.total_letters
        letter_to_words = letter_count / num_words
        words_to_sents = num_words / num_sentences
        return params["alw"] * letter_to_words + params["asw"] * words_to_sents - params["base"]

    def forcast(self, doc: Doc):
        """Returns the Forcast score for the document.
        """            
        params = self.measures["forcast"]
        sample_size = 150
        num_words = doc._.total_words
        eligible_words = [word for word in doc if not word.is_punct and not word.is_digit]
        if num_words < 150:
            return 0
        mono_syllabic = sum(1 for idx in random_sample(num_words, size=sample_size, replace=True)
                                if eligible_words[idx]._.syllables_count == 1)
        return params["base"] - (mono_syllabic / params["w"])

    def gunning_fog_index(self, doc: Doc):
        """Returns the Gunning Fog Index score for the document.
        """
        params = self.measures["gunning_fog_index"]
        words_to_sents = doc._.total_words / doc._.total_sentences
        # count words containing atleast 3 syllables
        complex_words = sum(1 for token in doc if (
            not token.is_punct
            and not token.is_digit
            and token._.syllables_count
            and token._.syllables_count >= 3))
        complex_words_proportion = complex_words / doc._.total_words
        return params["mult"] * (words_to_sents + 100 * complex_words_proportion)


def _get_num_syllables(doc: Doc, min_syllables: int = 1):
    """Return number of words in the document.
    Filters punctuation and words that start with apostrophe (aka contractions)
    """
    return sum(token._.syllables_count for token in doc if (
            token._.syllables_count
            and token._.syllables_count >= min_syllables
        )
    )
