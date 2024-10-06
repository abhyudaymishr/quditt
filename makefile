FILE=local/token.env
TOK=`cat $(FILE)`

build:
	python3 setup.py bdist_wheel sdist
	twine check dist/*

deploy:
	twine upload dist/* -u plutoniumm -p $(TOK)
	rm -rf build dist qudit.egg-info

test:
	pip install .
	python3 test.py