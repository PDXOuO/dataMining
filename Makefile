.PHONY: install predict check backtest figures report pledge all clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

predict:
	$(PY) scripts/predict_final.py

check:
	$(PY) run.py --check

backtest:
	$(PY) scripts/report_tables.py

figures:
	$(PY) scripts/make_figures.py

report:
	$(PY) scripts/make_report.py

pledge:
	$(PY) scripts/make_pledge.py

# 从零重建全部交付物
all: install predict figures backtest report pledge package

# 重建全部交付物并打包成可直接上传的 zip
package:
	$(PY) scripts/make_package.py

clean:
	rm -rf outputs/*.png outputs/result.csv data/cache/*.pkl
