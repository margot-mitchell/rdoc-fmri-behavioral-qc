# Targets CI runs: test, scan. The rest are local conveniences.
#
# `make hooks` must be run once per checkout: the pre-commit guard lives in
# .git/hooks/, which is not version controlled, so a fresh clone starts without it.

.PHONY: help test scan hooks demo wheel clean
.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

test: ## Run the test suite
	python3 -m pytest -q

scan: ## Identifier scan of the tracked tree, i.e. what would be published
	@rm -rf .scan_export && mkdir -p .scan_export
	@git archive HEAD | tar -x -C .scan_export
	python3 tools/scan_repo.py .scan_export --plain
	@rm -rf .scan_export

hooks: ## Install the pre-commit identifier guard
	cp tools/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	git config publicscan.tool "$(CURDIR)/tools/scan_repo.py"
	@echo "pre-commit guard installed"

demo: ## Run the whole pipeline on synthetic data and print the QC email
	python3 demo/run_demo.py

wheel: ## Build a wheel and list what it contains
	python3 -m pip wheel --no-deps --wheel-dir dist .
	@python3 -c "import zipfile,glob,collections; \
	  z=zipfile.ZipFile(sorted(glob.glob('dist/*.whl'))[-1]); \
	  c=collections.Counter(n.split('/')[0] for n in z.namelist()); \
	  [print(f'  {k}: {v} file(s)') for k,v in sorted(c.items())]"

clean:
	rm -rf .scan_export dist demo/_sandbox
