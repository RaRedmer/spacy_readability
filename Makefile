setup:
	poetry install
	poetry run python -m spacy download en_core_web_sm
	poetry run python -m spacy download de_core_news_sm

test:
	poetry run pytest --cov=spacy_readability -q --disable-pytest-warnings tests/

ci-test:
	poetry run pytest tests/ --cov=spacy_readability --junitxml=junit/test-results.xml
	poetry run codecov

lint:
	poetry run pylint spacy_readability

mypy:
	poetry run mypy spacy_readability

format:
	poetry run black spacy_readability tests