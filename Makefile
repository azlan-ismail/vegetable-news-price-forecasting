.PHONY: reproduce figures clean

reproduce:
	python src/reproduce_forecasting.py --check-reference

figures:
	python src/make_publication_figures.py

clean:
	rm -rf build
