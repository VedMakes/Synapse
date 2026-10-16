PYTHON ?= .venv/bin/python

.PHONY: test e1 e2 participant-maps participant-smoke

test:
	$(PYTHON) -m unittest discover -s tests -v

e1:
	$(PYTHON) tools/benchmark_vector_env.py --steps 10 --n-envs 1 2 4 8 16 --output reports/e1_vectorized_benchmark.md

e2:
	$(PYTHON) tools/run_e2_baseline.py --output reports/e2_baseline_convergence.md

participant-maps:
	$(PYTHON) tools/build_training_manifest.py

participant-smoke:
	$(PYTHON) tools/train_participant.py --steps 256 --budget-steps 10000 --output-dir participant_run
