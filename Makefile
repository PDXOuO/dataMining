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

# 把三样交付物打包到 submission/
package: predict report pledge
	mkdir -p submission
	cp outputs/result.csv "submission/提交结果_队伍名.csv"
	cp outputs/技术报告.pdf submission/
	cp outputs/承诺书.pdf submission/
	cp outputs/承诺书.docx submission/

clean:
	rm -rf outputs/*.png outputs/result.csv data/cache/*.pkl
