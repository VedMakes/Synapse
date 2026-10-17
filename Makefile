PYTHON ?= .venv/bin/python

.PHONY: test render e1 e2 participant-setup participant-maps participant-smoke participant-ui evaluate-participant

test:
	$(PYTHON) -m unittest discover -s tests -v

render:
	mkdir -p artifacts/replay
	$(PYTHON) tools/export_replay.py --scripted --track tracks/train/synthetic_oval.json --config config/simulation.json --steps 120 --participant-id pygame-smoke --output artifacts/replay/pygame-smoke.json
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy $(PYTHON) tools/run_pygame.py --replay artifacts/replay/pygame-smoke.json --headless

e1:
	$(PYTHON) tools/benchmark_vector_env.py --steps 10 --n-envs 1 2 4 8 16 --output reports/e1_vectorized_benchmark.md

e2:
	$(PYTHON) tools/run_e2_baseline.py --output reports/e2_baseline_convergence.md

participant-maps:
	$(PYTHON) tools/build_training_manifest.py

participant-setup:
	python3.11 tools/setup_participant.py

participant-smoke:
	$(PYTHON) tools/train_participant.py --steps 256 --budget-steps 10000 --output-dir participant_run

participant-ui:
	$(PYTHON) tools/participant_ui.py

evaluate-participant:
	$(PYTHON) tools/evaluate_participant.py --artifact-dir submission --track tracks/train/synthetic_oval.json --config config/simulation.json --repeats 2
