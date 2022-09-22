SRC := src

style:
	pip install -r ./requirements/style.txt
	isort $(SRC)
	black $(SRC) --line-length 79