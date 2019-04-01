.PHONY: install
install:
	pip3 install -Ue .

.PHONY: test
test:
	python ./score_splitter.py

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
