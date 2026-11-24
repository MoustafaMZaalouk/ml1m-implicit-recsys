PY ?= python3
export PYTHONPATH := src

.PHONY: install data test reproduce report
install:
	$(PY) -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -c "from recsys import data; print(data.ensure_data())"

test: data
	$(PY) -m pytest -q

report:
	$(PY) -m recsys.report

# One command: download + verify, run the whole ladder, regenerate figures/README tables, run tests.
reproduce: data
	$(PY) -m recsys.experiment
	$(PY) -m recsys.report
	$(PY) -m pytest -q
