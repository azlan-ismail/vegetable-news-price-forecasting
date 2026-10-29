.PHONY: reproduce figures diagnostics clean

reproduce:
	python src/reproduce_forecasting.py --check-reference

figures:
	python src/make_publication_figures.py

diagnostics:
	python src/diagnostic_reanalysis.py
	python src/clark_west.py
	python src/make_diagnostic_figures.py

clean:
	rm -rf build
