.PHONY: install
install:
	pip install -Ue .

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
