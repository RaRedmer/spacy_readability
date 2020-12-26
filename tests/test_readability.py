import spacy
import pytest

from spacy.tokens import Doc
from spacy_readability import (
    Readability,
    _get_num_syllables,
)


@pytest.fixture(scope="function")
def nlp():
    return spacy.load("en")


@pytest.fixture(scope="function")
def read():
    pipeline = spacy.load("en")
    return Readability(nlp=pipeline)


def test_simple(nlp):
    doc = nlp("sample")
    assert doc


def test_integration(nlp, read):
    nlp.add_pipe(read, last=True)
    assert "readability" == nlp.pipe_names[-1]


def test_sentences(nlp, read):
    nlp.add_pipe(read, last=True)
    doc = nlp("I am 2 sentences. I am the best panda?")
    assert doc._.total_sentences == 2


def test_words(nlp, read):
    nlp.add_pipe(read, last=True)
    doc = nlp("I contain four words.")
    assert doc._.total_words == 4


def test_syllables(nlp, read):
    nlp.add_pipe(read, last=True)
    doc = nlp("I contain four words.")
    for token in doc:
        print(token, token._.syllables_count)
    assert doc._.total_syllables == 5


def test_extensions(nlp, read):
    """ Values obtained by manual calculation.
    """
    nlp.add_pipe(read, last=True)
    doc = nlp("I contain four words. Therefore, it should be possible to calculate by hand.")
    syllable_result = {
        "i": 1,
        "contain": 2,
        "four": 1,
        "words": 1,
        "therefore": 2,
        "it": 1,
        "should": 1,
        "be": 1,
        "possible": 3,
        "to": 1,
        "calculate": 3,
        "by": 1,
        "hand": 1,
    }
    letter_result = {
        "i": 1,
        "contain": 7,
        "four": 4,
        "words": 5,
        "therefore": 9,
        "it": 2,
        "should": 6,
        "be": 2,
        "possible": 8,
        "to": 2,
        "calculate": 9,
        "by": 2,
        "hand": 4,
    }
    assert Doc.has_extension("flesch_kincaid_grade_level")
    assert Doc.has_extension("flesch_kincaid_reading_ease")
    assert Doc.has_extension("dale_chall")
    assert Doc.has_extension("smog")
    assert Doc.has_extension("coleman_liau_index")
    assert Doc.has_extension("automated_readability_index")
    assert Doc.has_extension("forcast")
    
    assert doc._.total_sentences == 2
    assert doc._.total_words == 13
    assert doc._.total_syllables == 19
    assert doc._.total_letters == 61

    assert syllable_result == {word.text.lower(): word._.syllables_count for word in doc if not word.is_punct and not word.is_digit}
    assert letter_result == {word.text.lower(): word._.letters_count for word in doc if not word.is_punct and not word.is_digit}
    # test extension values
    assert pytest.approx(4.69, rel=1e-2) == doc._.total_letters / doc._.total_words
    assert pytest.approx(1.46, rel=1e-2) == doc._.total_syllables / doc._.total_words
    assert pytest.approx(6.5, rel=1e-2) == doc._.total_words / doc._.total_sentences
    assert pytest.approx(4.19, rel=1e-2) == doc._.flesch_kincaid_grade_level
    assert pytest.approx(7.22, rel=1e-2) == doc._.coleman_liau_index
    assert pytest.approx(3.92, rel=1e-2) == doc._.automated_readability_index
    assert 0 == doc._.smog

# @pytest.mark.parametrize("text,expected", [("", 0), ("#", 0)])
# def test_edge_scenarios(text, expected, nlp, read):
#     nlp.add_pipe(read, last=True)
#     doc = nlp(text)
#     assert doc._.flesch_kincaid_grade_level == expected
#     assert doc._.flesch_kincaid_reading_ease == expected
#     assert doc._.coleman_liau_index == expected
#     assert doc._.automated_readability_index == expected
#     assert doc._.smog == expected
#     assert doc._.dale_chall == expected
#     assert doc._.forcast == expected
