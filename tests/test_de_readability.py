import spacy
import pytest
import numpy as np

from spacy.tokens import Doc
from spacy_readability import (
    Readability,
    _get_num_syllables,
)


@pytest.fixture(scope="function")
def nlp_ger():
    return spacy.load("de_core_news_sm")


@pytest.fixture(scope="function")
def read_ger():
    pipeline = spacy.load("de_core_news_sm")
    np.random.seed(123)
    return Readability(nlp=pipeline)


def test_simple(nlp_ger):
    doc = nlp_ger("Beispiel")
    assert doc


def test_integration(nlp_ger, read_ger):
    nlp_ger.add_pipe(read_ger, last=True)
    assert nlp_ger.pipe_names[-1] == "readability"


def test_sentences(nlp_ger, read_ger):
    nlp_ger.add_pipe(read_ger, last=True)
    doc = nlp_ger("I am 2 sentences. I am the best panda?")
    doc = nlp_ger("Ich bestehe aus zwei Sätzen. Wie viel wurden gezählt?")
    assert doc._.total_sentences == 2


def test_words(nlp_ger, read_ger):
    nlp_ger.add_pipe(read_ger, last=True)
    doc = nlp_ger("Ich bestehe aus fünf Wörtern.")
    assert doc._.total_words == 5


def test_syllables(nlp_ger, read_ger):
    nlp_ger.add_pipe(read_ger, last=True)
    doc = nlp_ger("Ich bestehe aus fünf Silben.")
    assert doc._.total_syllables == 8

@pytest.mark.parametrize("text,expected", [("", 0), ("#", 0)])
def test_edge_scenarios_ger(text, expected, nlp_ger, read_ger):
    nlp_ger.add_pipe(read_ger, last=True)
    doc = nlp_ger(text)
    # assert doc._.flesch_kincaid_grade_level == expected
    assert doc._.flesch_kincaid_reading_ease == expected
    assert doc._.gunning_fog_index == expected
    # assert doc._.coleman_liau_index == expected
    # assert doc._.automated_readability_index == expected
    # assert doc._.smog == expected
    # assert doc._.dale_chall == expected
    # assert doc._.forcast == expected
