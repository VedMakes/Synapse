PYTHON ?= .venv/bin/python

.PHONY: test e1 e2

test:
	$(PYTHON) -m unittest discover -s tests -v

e1:
	$(PYTHON) tools/benchmark_vector_env.py --steps 10 --n-envs 1 2 4 8 16 --output reports/e1_vectorized_benchmark.md

e2:
	$(PYTHON) tools/run_e2_baseline.py --output reports/e2_baseline_convergence.md
