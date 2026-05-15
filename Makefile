# M.I.N.E.R.V.A. — common developer commands
# Run `make help` to see available targets.

.PHONY: help test test-quick test-verbose lint clean install install-dev pipeline-templates pipeline-full

help:
	@echo "M.I.N.E.R.V.A. — Available make targets"
	@echo ""
	@echo "  make test              Run full test suite (271 expected)"
	@echo "  make test-quick        Quiet test output"
	@echo "  make test-verbose      Verbose with full tracebacks"
	@echo ""
	@echo "  make install           Install runtime deps (pip install -r requirements.txt)"
	@echo "  make install-dev       Install dev deps (lighter, for testing)"
	@echo ""
	@echo "  make pipeline-templates  Run templates-only pipeline (no GPU, ~5 min)"
	@echo "  make pipeline-full       Run full pipeline (GPU recommended, ~30 min)"
	@echo ""
	@echo "  make clean             Remove caches, generated/, logs/ (keeps models/)"

test:
	python -m pytest tests/ -v

test-quick:
	python -m pytest tests/ -q

test-verbose:
	python -m pytest tests/ -v --tb=long

install:
	pip install -r requirements.txt

install-dev:
	pip install -r dev-requirements.txt
	pip install pytest pydantic

pipeline-templates:
	@echo "Running templates-only pipeline (no GPU)..."
	python scripts/30_template_scenario_generator.py
	python scripts/31_universal_pseudonymize.py
	python scripts/35_pseudonymize_places.py
	python scripts/21_balance_unity_cards.py
	python scripts/23_enforce_election_theme.py
	python scripts/24_curate_teaching_cards.py
	python scripts/33_strict_name_allowlist.py
	python scripts/26_faithfulness_audit.py
	python scripts/28_draw_user_deck.py
	@echo "Done. Outputs in generated/ and reports/."

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf generated/*
	rm -rf logs/*
	@echo "Cleaned. Models/, data/, templates/ preserved."
