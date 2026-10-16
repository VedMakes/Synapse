# SYNAPSE participant delivery options

**Research date:** 2026-10-17
**Question:** What is the best way to deliver a fixed Python training environment to college participants and receive models for the SYNAPSE reinforcement-learning event?
**Scope:** Public GitHub repository; public/partial participant repository; standalone executable; Python wheel/source package; Docker/Dev Container; and hybrid designs. Criteria are reproducibility, cross-platform support, offline use, trust/cheat resistance, setup friction, update/versioning, licensing, and organizer operational burden.
**Source policy:** Primary documentation was prioritized. Web sources below were checked on 2026-10-17; where a source exposes a publication/update date, it is recorded. Claims marked **Inference** are event-specific engineering conclusions rather than statements made by the cited source.

## Executive recommendation

Use a **hybrid with a public, versioned starter repository and a model-only submission contract**:

1. **Publish a public starter kit** for the participant-facing training path: source, README, fixed environment API, example trainer, public development tracks, tests, deterministic smoke evaluation, and an exact release/tag. A participant should be able to clone/download one release and train without depending on an online service.
2. **Make the source repository the canonical reference**, not an executable. Publish a matching source archive and, if convenient, a Python wheel for the stable `racing_event` import API. Do not make participants build the wheel unless they need to develop the package.
3. **Provide an offline release bundle**: the source archive, the wheel, a platform-appropriate wheelhouse for the fixed Python/OS targets, a hash manifest, licenses/notices, and a short install script or exact commands. The existing lockfile is useful, but a lockfile containing index URLs is not by itself an offline artifact; the wheels must also be present locally.
4. **Offer Docker/Dev Container as an optional reproducibility path**, primarily for maintainers, CI, advanced participants, and a known-good local environment. Pin the base image by digest and publish the exact image digest. Do not require Docker for a first-time college participant unless the event can support Docker Desktop/Engine installation and virtualization issues.
5. **Submit a model artifact, not participant execution code, for the default leaderboard.** For this project’s planned ONNX boundary, accept an ONNX model plus a small JSON manifest and evaluate it with the organizer-owned Python simulator/evaluator on organizer-owned infrastructure. Keep held-out tracks and scoring code off the participant machine. The evaluator should validate the graph, operator set, tensor shapes, file size, and model metadata before running it.
6. **If arbitrary participant code is a deliberate educational objective**, offer a separate code-submission division. Run that code only inside an organizer-owned, fixed image and sandbox with no network, read-only/private mounts, CPU/memory/time/process limits, and a fresh worker per submission. Do not let participants provide Dockerfiles or images to the evaluator in the ordinary division.

This balances accessibility and fairness: students see and can modify the learning environment, while the scoring path is a small, versioned artifact interface rather than an arbitrary program. No packaging choice alone prevents cheating. Hidden evaluation data, server-side scoring, resource limits, provenance checks, and clear rules do the anti-cheat work.

### Short decision matrix

Ratings are relative for a college event whose target is Python 3.11, local/offline training, and server-side model evaluation. “High” is favorable for the criterion except where the heading says “burden” or “friction.”

| Delivery method | Reproducibility | Cross-platform | Offline | Trust / cheat resistance | Participant setup friction | Versioning | Organizer burden | Recommended role |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full public GitHub repository | Medium–high when pinned to a commit/tag; lower from a moving branch | High for source, dependent on Python/native wheels | Medium; source is offline, dependencies may not be | Low for secrecy; high transparency, but public code/tracks are inspectable | Low–medium | High with tags/releases; weak if instructions say “clone main” | Low–medium | Canonical starter kit and public development environment |
| Partial/public participant repository | High if the public boundary is versioned and private evaluator is separately pinned | High for the public source path | High with an offline bundle | High relative to the alternatives, because held-out data/evaluator stay private | Medium; more rules and boundary documentation | High, but requires synchronized public/private release IDs | Medium–high | **Best default architecture** |
| Standalone executable | Medium for one target build; weak across OS/CPU matrices | Low–medium; separate build per OS/Python/architecture | High once downloaded | Low; executable is arbitrary code and packaging is not a sandbox; PyInstaller bundles bytecode that can be reverse-engineered | Low for users after download, high for organizers | Medium; binary release management and signing required | High | Optional convenience launcher, not primary environment or submission |
| Python wheel/source package | High for a tagged package plus pinned dependencies; wheel is a distribution, not a full environment | High for pure Python; native dependencies need matching wheels/platform tags | High with a wheelhouse; low if it points to PyPI | Low if the server installs participant packages; good if only organizer code is installed; source is inspectable | Medium; Python/venv/pip knowledge required | High via immutable versions and release files | Medium | Stable API package and offline release component |
| Docker image / Dockerfile | High when image digest, build inputs, and limits are fixed | Medium; Linux container needs Docker/VM and architecture-specific images | High after image pull; very high with `docker save`/`load` bundle | High for **organizer-owned image + model artifact**; medium for arbitrary code; containers are isolation, not an absolute security boundary | Medium–high, especially on managed Windows/macOS machines | High with immutable digests, tags, SBOM/attestation | High | Optional dev environment; fixed evaluator image; code-submission worker |
| Dev Container | High for recreating development tools when implementation supports the spec | Medium–high among Docker-capable tools | Medium–high if image/features are cached or bundled | Low as a submission mechanism; it is development metadata, not a security boundary | Medium–high: editor integration plus Docker | High when config/image is pinned | Medium | Convenience layer over Docker, not a participant artifact by itself |
| Hybrid: public repo + package + optional container + model-only submit | **High** | **High** if source/wheel supports target platforms and container is optional | **High** with bundled wheels/image | **High** for model-only server evaluation | **Low–medium** because a simple path exists | **High** with one release ID across all artifacts | Medium–high, but bounded by explicit artifacts | **Recommended** |

The table separates **delivery** from **evaluation**. A public repository, wheel, or container can deliver the training environment; none should be trusted as the scoring authority when participants can alter them.

## Event assumptions and current-project fit

The repository context reviewed on 2026-10-17 is already close to the recommended shape:

- `.python-version` pins Python 3.11 and `requirements.lock` pins the main dependency set, including CPU PyTorch.
- `README.md` describes a Python-authoritative simulator, deterministic evaluation, a future ONNX submission boundary, held-out-track planning, and an offline replay preview.
- `pyproject.toml` already has a setuptools build backend and package metadata.
- `Dockerfile` provides a baseline container path.

Important release observations for the event contract:

- The current `requirements.lock` contains `-e .`. That is convenient for local development but is not the best participant release primitive. An event bundle should install a versioned wheel or source distribution from a local `wheelhouse`, not an editable path that depends on the checkout layout.
- The project metadata says `requires-python = ">=3.10"`, while the README and `.python-version` identify Python 3.11 as the supported runtime. The public event release should make this relationship explicit: either support/test each declared Python version or narrow the event contract to the tested Python 3.11 target.
- The current Dockerfile uses the mutable `python:3.11-slim` tag and downloads dependencies during build. For a reproducibility release, record the exact base image digest and build from a local wheelhouse or publish the resulting image digest. These are notes for a future release process; no application code is changed by this research.
- The current project’s planned ONNX boundary is preferable to accepting PyTorch pickle/state-dictionary files from untrusted participants. PyTorch’s serialization documentation says `torch.save`/`torch.load` use Python pickle by default and warns never to load untrusted data; even `weights_only=True` narrows rather than eliminates denial-of-service/memory-corruption concerns. A model-only graph format with a fixed runtime gives the evaluator a smaller, more inspectable interface. [PyTorch serialization semantics, created/updated 2026-05-08](https://docs.pytorch.org/docs/2.14/notes/serialization.html) [PyTorch `torch.load` warning](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)

## What “best” means for this event

### Reproducibility is a release identity, not just a file format

To reproduce a race, the organizer needs one immutable **event release ID** covering:

- Python major/minor version and supported OS/architecture;
- environment package version and dependency hashes;
- observation schema and action schema versions;
- physics, reward, termination, time-step, and evaluation configuration;
- public training-track hashes and private held-out-track hashes held by the organizer;
- deterministic seed/start/evaluation procedure;
- evaluator image/package identity;
- model format, ONNX IR/opset policy, input/output names and shapes; and
- the exact submission hash and evaluation result.

A Git commit, wheel filename/version, Docker image digest, and model SHA-256 should all be recorded in the event result. A filename such as `final_model.onnx` is not a version identity.

Python’s own `venv` documentation is explicit that virtual environments are disposable and not movable/copyable: they should be recreated from requirements, and scripts can contain absolute interpreter paths. Therefore, never distribute `.venv` itself. Distribute the recipe plus local wheel files, or a container image.

## Option analysis

## 1. Full public GitHub repository

### Shape

Publish the complete participant training repository, for example:

```text
participant-repo/
  README.md
  LICENSE
  pyproject.toml
  requirements.lock
  .python-version
  racing_event/          # public simulator/API/trainer
  tracks/train/          # public training tracks
  examples/              # baseline and submission conversion
  tests/                 # public contract and smoke tests
  Dockerfile             # optional, pinned release path
  .devcontainer/         # optional development convenience
```

Students fork or download a tagged release, modify training code, and create a model artifact. The evaluator is not imported from the participant checkout; it is run from the organizer release.

### Strengths

- Lowest conceptual friction for a college audience: familiar Git, visible issues, pull requests, examples, and history.
- Excellent transparency. Students can inspect the environment, reproduce a baseline, add instrumentation, and learn software engineering practices.
- Source archives can be downloaded without Git history. GitHub documents that a commit archive has stable extracted file contents when the commit remains in the repository; branch/tag archives can change if the reference moves. Use immutable commit IDs or protected release tags for event instructions.
- GitHub gives a natural public changelog and release identity. A release should include the source archive, wheel, hashes, and installation notes rather than telling participants to use a moving branch.

### Weaknesses and risks

- A public repository exposes every public training track, public reward path, model contract, baseline, and implementation detail. That is acceptable for an open educational event, but it does not provide secrecy or prevent track-specific overfitting.
- Public source is not a security boundary. If participants submit a repository or a runnable script and the server executes it, they can alter the evaluator interface, access files, make network requests, consume unbounded resources, or fake local results unless the organizer runs a separate fixed evaluator.
- Dependency installation still depends on the participant’s Python, OS, architecture, network, compiler, and package indexes unless a wheelhouse or container is also released.
- Git history may contain more than the current tree. Do not publish secrets, private tracks, answer keys, internal evaluator code, or accidental credentials and assume that deleting them later makes them undiscoverable. Use a clean public repository boundary before launch.

### Best use

Use a full public repository as the **canonical learning and local-training interface** when the event values openness more than secrecy. It is especially good for a first pilot and public development-track phase. Pair it with a model-only submission contract if the leaderboard must be fair.

### Versioning rule

Instructions should say “download release `event-2026-v1`” or “checkout commit `<full hash>`,” never “clone `main`.” If a release fixes a bug in the training environment, issue a new release ID; do not silently alter a tag that has already been used for training or submissions.

### Licensing

Include an explicit repository license and third-party notices. Public visibility does not itself grant students permission to reuse, modify, or redistribute code. PyPA’s licensing guidance says that a project without a license generally gives users no permission to use or improve it; the event should choose a permissive license for the participant-facing starter code if that matches institutional policy. Separately state ownership and permitted use of tracks, logos, recordings, and participant submissions.

## 2. Partial/public participant repository

### Shape

Publish only what participants need to train:

```text
public participant release
  public simulator API and trainer boundary
  public development tracks
  examples/tests/docs
  conversion/validation client
  no held-out geometry
  no private scoring code, answer keys, signing secrets, or leaderboard logic

organizer release (private or controlled)
  same public release ID
  held-out tracks
  evaluator and scoring policy
  resource limits and sandbox configuration
  submission intake/storage/audit
```

The public participant repository can still be fully open source. “Partial” means the event’s operational system has a private evaluator boundary, not that the public starter code must be intentionally opaque.

### Strengths

- Best secrecy/fairness trade-off. Participants can reproduce local training while private tracks and scoring remain unavailable to normal participants.
- Matches competition practice. Kaggle’s host CLI distinguishes a public participant data bundle from a private rerun-scoring data bundle. Codabench runs participant submissions in a competition-specific organizer image and keeps reference data unavailable during the prediction step. AWS DeepRacer on AWS separates model upload/validation, evaluation workflows, queues, private data, and public leaderboards.
- Allows the public simulator and private evaluator to share a schema without exposing the implementation that chooses held-out tracks, retries, tie-breaks, or final ranking.

### Weaknesses and risks

- More documentation burden. Students must know which release is authoritative, what is intentionally private, which local results are only advisory, and what can be included in a submission.
- The public environment can still reveal enough behavior to overfit public tracks. Hidden tracks reduce direct memorization but do not prove generalization; use multiple held-out tracks, deterministic seeds, and a published evaluation protocol.
- Two release pipelines can drift. Every private evaluator should record the public API/schema release ID and reject incompatible submissions. Add a compatibility smoke test that runs the public baseline through the private evaluator before opening submissions.
- A hidden evaluator is not a license to make the public task ambiguous. Keep input/output shapes, action bounds, termination semantics, file limits, and failure messages documented.

### Best use

This is the **recommended event structure**. Publicly release the training environment and model conversion client; privately retain only what must be private: held-out tracks, final scoring, queue/worker configuration, and secrets. If the educational goal includes inspecting the full simulator, publish that simulator source; secrecy should reside in test data and evaluation orchestration rather than unexplained binaries.

### Versioning rule

Use one release manifest, for example:

```json
{
  "event_release": "2026.1",
  "participant_api": "obs-action-1.2",
  "public_package": "racing-event==0.1.0",
  "evaluator_image_digest": "sha256:...",
  "model_format": "onnx",
  "held_out_track_set": "private-hash-not-published"
}
```

Do not put the private track hash if the hash itself is considered sensitive; instead record it in the organizer audit record. Publish the public release manifest and keep a signed internal counterpart.

### Licensing

License the public starter code and list all third-party licenses. Keep private evaluator code under the institution/organizer’s chosen terms. The event rules should grant the organizer a limited license to store, execute, audit, and display submitted model metadata/results, while leaving participant ownership of original training code and model weights clear. A participant model can also carry a model license in ONNX metadata; do not infer a license from the source repository’s license.

## 3. Standalone executable

### Shape

Package the Python launcher/trainer/simulator using PyInstaller or a similar freezer and distribute one executable or a platform-specific folder. Participants run it directly, perhaps with a configuration file and a mounted training directory.

### Strengths

- No separate Python installation or pip dependency resolution for the participant.
- Good offline launch experience after download.
- Convenient for a short event-day demo, baseline viewer, replay viewer, or a small submission-conversion utility.
- Can reduce accidental edits to the organizer’s launcher and make “click to run” materials approachable.

### Weaknesses and risks

- PyInstaller’s official documentation says output is specific to the active OS and Python version; a separate build is needed for each OS, Python version, and 32/64-bit target. This creates a matrix such as Windows x64, macOS arm64, macOS x64, and Linux x64, with separate smoke tests and artifact hosting.
- Native dependencies, BLAS/OpenMP behavior, GPU libraries, antivirus/quarantine, code signing, macOS notarization, Windows permissions, and Linux `glibc` compatibility add failure modes. PyInstaller documents that Linux bundles do not bundle `libc` and should be built on the oldest supported Linux release.
- It is not source protection. PyInstaller says the bundle contains compiled Python bytecode that can in principle be decompiled. Obfuscation is not a fair anti-cheat control and can make debugging harder.
- An executable still runs arbitrary participant-controlled behavior if it is the submission. A binary signature proves who built a file, not that it cannot inspect files, use the network, or exploit evaluator assumptions.
- A one-file bundle extracts libraries to a temporary directory at startup; PyInstaller documents `noexec` failures, leftover extraction directories after crashes, and slower startup. Large PyTorch bundles also create substantial download/storage cost.

### Best use

Use a standalone executable only for a **launcher, replay viewer, local validator, or optional baseline**. It is a poor canonical delivery format for a Python RL training environment and a poor default model-submission format.

### Versioning and release process

Build each target from a clean, pinned environment; include a build manifest, SHA-256, supported OS/architecture, and a visible `--version` output. Keep the executable as a convenience artifact corresponding to the same event release ID as the source/wheel/container. Do not silently replace an executable at the same URL.

### Licensing

Audit bundled Python/native dependencies and include their notices. Code signing certificates, notarization identities, and signing keys must stay in the release system, never in the public repository. A signed binary does not remove the need for open event rules and source licensing.

## 4. Python wheel and source package

### Shape

Build the participant environment/API as a normal Python distribution:

- `racing-event-<version>-py3-none-any.whl` for pure Python code/data where applicable;
- a source distribution (`.tar.gz`) for inspectability and fallback builds;
- a pinned requirements/lock file; and
- a directory of all required dependency wheels for offline installation.

The current project already uses `pyproject.toml` and setuptools, so this is a natural release path for the stable package API.

### Strengths

- Standard Python tooling and metadata. PyPA documents that `pyproject.toml` declares the build backend, project name/version, Python requirement, dependencies, scripts, and license metadata.
- A wheel installs faster than an sdist because it is already built. A pure-Python wheel can be broadly portable; platform wheels advertise compatibility through Python/ABI/platform tags.
- Immutable package versions are a clean update mechanism. Publish `0.1.0`, `0.1.1`, or an event-specific version and never mutate an already-used artifact.
- Good offline support. Pip documents wheelhouses: pre-build/download all wheels, transfer an archive, then install with `--no-index --no-deps --find-links ...`. This avoids package-index access and avoids compilation at participant setup time.
- Easier to test than a frozen executable: import the package, run the contract tests, and inspect its metadata. Source packages are particularly useful in a college course where students need to understand and modify the simulator/trainer.

### Weaknesses and risks

- A wheel is not a complete Python runtime. Participants still need the right Python interpreter and compatible native wheels for NumPy, PyTorch, BLAS/OpenMP, and their OS/architecture. A “universal” package can therefore hide a non-universal dependency problem.
- A source distribution may invoke a build backend and compiler and may resolve dependencies from an index. That is unsuitable as the only offline artifact.
- Pip’s version pinning alone is weaker than hashes plus local artifacts. Pip’s repeatable-install documentation says pinned versions work across OSes/architectures but trust the fetch locations; hash-checking protects against changed/compromised downloads, while wheelhouses improve availability but can be OS/architecture-specific.
- Installing a participant-provided package into the evaluator is executing untrusted installation/import code. Do not accept “upload your wheel” as a security boundary. Use the wheel for organizer-controlled code, or execute participant code only in a sandboxed worker.
- Publishing the whole simulator as a package does not conceal public tracks or private logic. A package on a public index is also an unnecessary long-term namespace and dependency-management commitment for a one-off event; a GitHub release asset or private index can be simpler.

### Best use

Use a wheel/source package as the **stable installable component inside the hybrid release**. Make the simplest supported path a local wheelhouse installation in a fresh Python 3.11 venv. Keep the source tree available for learning and modification; install the public package non-editably for a clean release test.

### Recommended offline installation shape

The release should contain, for each supported target (at minimum the event’s actual target OS/architecture):

```text
event-2026.1/
  participant-source.tar.gz
  racing_event-0.1.0-py3-none-any.whl
  wheelhouse/
    racing_event-0.1.0-py3-none-any.whl
    numpy-...whl
    torch-...whl
    ...
  requirements-offline.txt       # hashes, local-compatible artifacts
  SHA256SUMS
  LICENSE
  THIRD_PARTY_NOTICES/
```

Participant setup is then conceptually:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse \
  --require-hashes -r requirements-offline.txt
```

The exact command must be tested on every supported platform. A wheelhouse built on Linux cannot automatically serve Windows/macOS native dependencies; publish one per compatibility target or constrain the event to a tested environment.

### Versioning and licensing

Use PEP 440-compatible versions and put the event release ID in package metadata or a companion manifest. Include an SPDX license expression and license files for the public package and third-party components. PyPA’s current guidance (PEP 639) says license files and expressions should be included in distribution metadata; if vendored code has multiple licenses, retain all required notices.

## 5. Docker image and Dockerfile

### Shape

Publish a Dockerfile and/or prebuilt image that contains the fixed Python environment. A participant installs Docker, pulls the image, mounts a training workspace, and runs the trainer. The evaluator uses an organizer-owned image and receives a model artifact.

### Strengths

- Strong environment capture: OS userspace, Python interpreter, native libraries, package versions, and command entrypoint can be tested together.
- A fixed image digest is a stronger identity than a mutable tag. Docker’s build guidance says tags are mutable and recommends digest pinning for exact image content, while also noting that pinning trades away automatic security updates; use a controlled update process with a new digest/release ID.
- Good offline transport after the image is available. Docker’s official `image save` creates a tar archive containing image layers/tags, and `image load` restores images/tags from tar or compressed input. Supply image archive hashes for an event-day network-independent setup.
- Server-side use with an organizer-owned image gives a stable evaluator and allows CPU/memory constraints, no-network configuration, non-root execution, and a fresh container per run.
- Codabench is a concrete competition precedent: it runs participant submissions in a competition-specific Docker image, and its compute worker creates a submission container with organizer-controlled directories. AWS DeepRacer on AWS likewise uses containerized model validation/optimization and isolated private-subnet validators.

### Weaknesses and risks

- Docker is not “install-free.” Linux users need Docker Engine/Podman permissions; Windows/macOS users generally need a VM-backed desktop product, disk space, virtualization, and campus IT compatibility. Docker’s current Desktop documentation also describes subscription terms for larger commercial enterprises; a college event should verify institutional usage terms rather than assume every machine can install it.
- Image downloads can be very large, especially with PyTorch. Building from a Dockerfile offline requires all base layers and package artifacts already cached or transferred; a Dockerfile alone is not an offline bundle.
- Cross-platform means “same Linux userspace under a supported container engine,” not identical native behavior. CPU architecture, hardware acceleration, filesystem mounts, line endings, and Docker Desktop virtualization can still affect performance.
- Docker’s own security documentation says only trusted users should control the Docker daemon because host filesystem sharing can let a container alter the host. Never give a participant submission access to the host Docker socket or privileged mode. Containers use namespaces/cgroups and dropped capabilities, but they are not a substitute for a hardened worker/VM boundary.
- Reproducible image builds need more than a pinned `FROM`: package indexes, apt package versions, build timestamps, source tree, and generated artifacts must be controlled. Docker documents `SOURCE_DATE_EPOCH` support for image metadata; use it in CI where byte-level reproducibility matters.

### Best use

Use Docker in three bounded roles:

1. **Optional participant development path** for advanced users who want a one-command environment.
2. **Organizer-owned evaluator image**, pinned by digest and used for all leaderboard scoring.
3. **Separate code-submission sandbox**, if the event explicitly accepts arbitrary code.

Do not make the default participant submission be a Docker image. Asking each student to build and upload an image makes image supply-chain, storage, scheduling, and sandbox risks the organizer’s problem without improving a model-only leaderboard.

### Versioning and operational rules

- Publish both a human tag (`event-2026.1`) and immutable digest.
- Keep the Dockerfile, build context hash, base image digest, dependency lock, SBOM, and test report in the organizer release record.
- Run as a non-root user; do not mount private tracks writable; disable network for evaluation unless there is a documented need; cap CPU, memory, process count, wall time, and output size.
- Use a separate worker/VM or hardened container runtime for untrusted code. A normal developer laptop Docker daemon is not an appropriate evaluator for arbitrary submissions.
- Keep security updates in a documented cadence. A security rebuild is a new evaluator release and must not change the scoring contract mid-round.

### Licensing

The Dockerfile/image must preserve the licenses and notices of Python, OS packages, Python dependencies, and any copied source. Image labels can record version and license information, but labels do not replace the license files or notices. A private registry may simplify distribution but adds credentials, retention, access-control, and availability work; a public registry is simpler but exposes the image and its layers.

## 6. Dev Container

### Shape

Add `.devcontainer/devcontainer.json` that points to the Dockerfile or a prebuilt image and gives an editor a reproducible development environment. The Development Container Specification describes `devcontainer.json` as metadata for creating repeatable development environments; it supports image, Dockerfile, Compose, lifecycle commands, mounts, users, and tool customizations.

### Strengths

- Excellent onboarding for VS Code/GitHub Codespaces-compatible workflows: clone, reopen in container, run tests.
- Separates development metadata from the deployment image concept. The spec explicitly notes that development containers may contain tools that should not be in a deployment image and may use different secrets/settings.
- Can point at the same pinned Docker image used for CI, reducing “works on my machine” problems.

### Weaknesses and risks

- It is not a new runtime format and it is not a security boundary. A Dev Container is an instruction set for a tool that creates containers; the underlying engine, image, features, mounts, and lifecycle scripts still matter.
- It adds Docker and editor-specific friction to a population that may only need `python -m venv` and a local wheelhouse.
- Dev Container Features and lifecycle scripts can add network dependencies and mutable installation behavior. Pin images/features or avoid them in the event-critical path.
- A development container generally mounts the participant workspace. That is appropriate for editing but not an evaluator configuration; never use the participant workspace as the source of private evaluator code or tracks.

### Best use

Offer it as a **quality-of-life layer** over the public repository and Docker image. Document it after the native Python/offline path, not before it. Keep the authoritative contract in ordinary files and commands so participants are not locked to one editor.

## 7. Hybrid designs

### Recommended participant-facing release

**Tier A — Native/offline path (default):**

- public GitHub release/tag and source archive;
- Python 3.11 requirement;
- versioned wheel and source package;
- local wheelhouse with exact dependency artifacts and hashes;
- public tracks and examples;
- `make test`/equivalent contract smoke test;
- deterministic local evaluation and model export/validation command;
- `SHA256SUMS`, licenses, third-party notices, and a release manifest.

**Tier B — Container path (optional):**

- matching Dockerfile and prebuilt image digest;
- optional Dev Container metadata;
- image tar archive for event-day/offline delivery;
- the same public release ID and the same test/evaluation command.

**Tier C — Submission path (default leaderboard):**

- ONNX model file;
- manifest containing event release ID, schema ID, input/output names and shapes, model SHA-256, producer/version, and optional author/license metadata;
- no Python source, wheels, Dockerfile, executable, network dependency, or participant callback at scoring time.

**Tier D — Optional code division:**

- source archive or a constrained entrypoint bundle;
- organizer image only;
- no participant-supplied base image or Docker socket;
- sandboxed worker and a separate score/ruleset.

### Why this beats a single format

- GitHub gives transparency and teaching value.
- Wheelhouse gives offline, repeatable native setup.
- Docker gives maintainers and advanced participants a stronger environment capture.
- Model-only ONNX gives the evaluator a small, inspectable, cross-language artifact boundary.
- A private evaluator/held-out-track set preserves competition integrity.
- No student is forced to learn Docker, packaging, or Git beyond what the event’s educational goal requires.

### Hybrid failure mode to avoid

Do not publish five independently versioned ways to install the environment. Every artifact must identify the same event release, and all paths must run the same contract tests. If native Python, wheel, Docker, and executable paths produce different observation/action behavior, the event has multiple tasks rather than multiple delivery methods.

## Trust, cheating, and submission policy

### Fundamental boundary

If the scoring process executes participant-controlled Python, a participant-controlled executable, a participant-controlled wheel install, or a participant-controlled Docker image, assume the submission is untrusted. File hashes establish integrity after upload; they do not make code safe. A checksum in a replay or model manifest also proves identity, not anti-cheat authenticity, unless it is bound to a trusted upload/audit system.

### Preferred model-only pipeline

1. Receive an upload over an authenticated submission channel.
2. Record participant/team, timestamp, event release ID, original filename, byte count, and SHA-256.
3. Enforce maximum upload size and reject archives/extra files if the contract allows one model file.
4. Parse and validate ONNX using the official checker and an allowlist of IR/opset versions, operator domains, input/output names, ranks, dtypes, and dimensions. Reject custom operators, external tensor paths, unexpected graph outputs, training graphs, or oversized tensors unless deliberately supported.
5. Store the original artifact immutably and create a normalized audit record. Do not rewrite a model and then lose the submitted bytes.
6. Load it in the organizer-pinned inference runtime with fixed session options. Disable network and filesystem access from the evaluator process where possible; do not load custom operator libraries.
7. Run fixed-start, deterministic evaluation across private tracks/seeds, with CPU/memory/wall-time/output limits and a fresh process/container per submission.
8. Repeat a selected evaluation to verify deterministic trajectory/result hashes. Record per-track outcomes, DNF reason, runtime, resource use, and evaluator identity.
9. Publish only the permitted score/telemetry. Keep private track geometry, answer keys, and internal failure details private.

ONNX’s IR specification describes an inference model as a serialized computation graph and says an inference model without random-generator operations is expected to produce the same output on the same input. It also defines versioned IR/opsets, metadata including `model_license`, and a checker. These properties support validation, but ONNX is not automatically safe: it is extensible, supports external tensor data and custom operator domains, and can consume substantial resources. The event must define a narrow supported subset and reject everything else.

### If code submission is required

Use an organizer-owned fixed image and evaluator wrapper. The participant supplies only a documented entrypoint/source bundle, not an image or daemon configuration. Apply:

- no network or only an allowlisted local IPC channel;
- no host mounts, Docker socket, privileged mode, or host PID/IPC namespaces;
- read-only application and private-track mounts;
- non-root user and reduced capabilities;
- CPU, memory, disk, process, file-descriptor, wall-time, and output limits;
- fresh worker/container/VM per submission;
- deterministic seed and fixed environment variables;
- explicit allowed files and working directory;
- logging and termination reason;
- malware/secret scanning and archive traversal protections; and
- a manual appeal/re-run procedure with the exact artifact and evaluator digest.

Codabench’s documentation is a useful operational precedent: it accepts a zip bundle, distinguishes code submissions from result-only submissions, runs code in a competition-specific organizer image, provides controlled input/output directories, and executes the scoring workflow on compute workers. AWS DeepRacer on AWS goes further for a production service: its current public source describes model validation in private isolated subnets with no outbound traffic, malware scanning for physical models, queues, and separate evaluation workflows. These are examples of the organizer owning the execution boundary, not evidence that a student Docker image is safe to run directly.

## Competition practice relevant to the decision

### Kaggle

The official Kaggle CLI repository and documentation show several useful patterns:

- Ordinary competitions upload a submission file, then poll status/score; a code competition can associate the submission with a versioned notebook/kernel.
- Competition data is versioned; host commands update data with a message and distinguish the public data bundle from private rerun-scoring data.
- Hosts can stage pages before publishing, freeze a private solution file before launch, and set runtime caps/leaderboard settings.
- Model variation versions are uploaded as versioned files, which is a good precedent for treating model artifacts as immutable versions rather than overwriting one filename.

This favors a simple upload-and-score model artifact for the default division and an explicit, versioned code runtime for a separate code division. It also supports keeping public training data and private final-scoring data as separate release objects.

Primary sources (accessed 2026-10-17):

- Kaggle CLI repository, Apache-2.0: https://github.com/Kaggle/kaggle-cli
- Competition commands: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competitions.md
- Competition creation/versioned public vs rerun data: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competition_creation.md
- Submission/tutorial workflow: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/tutorials.md

### Codabench/CodaLab practice

Codabench’s official documentation is directly relevant to benchmark submissions:

- Participants upload a `.zip` bundle. A submission can be code (with metadata specifying a command) or results only (no code executed on the platform).
- The organizer specifies the competition Docker image. A compute worker creates a separate submission container and exposes controlled directories for input, output, ingestion, scoring, reference data, and participant code.
- The site queues work to compute workers and reports results after scoring. Participant routing can send groups to distinct compute queues, which is useful if a college event has CPU and GPU cohorts.
- Competition bundles and task specifications are separately managed from participant submissions.

The operational lesson is strong: **a competition can accept a compact result artifact when code execution is unnecessary, and when code is necessary the organizer owns the image and worker layout**.

Primary sources (accessed 2026-10-17; current docs do not expose a page publication date):

- Participants and result/code zip submissions: https://docs.codabench.org/latest/Participants/User_Participating-in-a-Competition/
- Organizer competition image: https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Competition-docker-image/
- Submission process/compute worker: https://docs.codabench.org/latest/Developers_and_Administrators/Submission-Process-Overview/
- Submission container directories and reference-data boundary: https://docs.codabench.org/latest/Developers_and_Administrators/Submission-Docker-Container-Layout/
- Participant routing to compute queues: https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Participant-Routing/
- Codabench source repository, Apache-2.0: https://github.com/codalab/codabench

### AWS DeepRacer on AWS practice

The current AWS DeepRacer on AWS implementation guide says its publication date is September 2026. Its public repository describes a self-hosted RL racing system with virtual races, imported models, evaluation jobs, physical model export, multi-track events, queues, leaderboards, and separate model validation/optimization components.

The most relevant architecture points are:

- uploaded model assets are stored separately from user data and evaluation workflows;
- model import validates/sanitizes assets before they enter the system;
- model validators and reward validators run in private isolated subnets with no outbound traffic;
- GuardDuty malware scanning is used for physical model uploads;
- jobs are queued and run through fixed workflows, rather than trusting a browser/local score;
- live races process queued submissions one at a time and record rankings/results; and
- public leaderboards are derived from server-side race state.

This is a large cloud system and should not be copied wholesale for a college event. It is evidence for the boundary pattern: upload a model, validate it, run it in organizer-owned infrastructure, and record a reproducible result. It also demonstrates that a model submission system’s operational burden can grow substantially once accounts, queues, private data, malware scanning, physical deployment, and live leaderboards are included.

Primary sources (accessed 2026-10-17):

- AWS implementation guide, publication date September 2026: https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/
- Public AWS solution repository: https://github.com/aws-solutions/deepracer-on-aws
- Repository README/architecture and model validation workflow: https://github.com/aws-solutions/deepracer-on-aws/blob/main/README.md

## Criteria in detail

### Reproducibility

**Best:** organizer-owned fixed evaluator image/digest plus model-only artifact; public release ID ties it to the training contract.
**Good:** wheel/source plus exact Python version, hashes, and wheelhouse.
**Conditional:** GitHub source pinned to a commit/tag; source alone does not freeze the interpreter/native libraries.
**Conditional:** executable for a single target, but cross-platform build differences and native OS libraries remain.
**Weak if unpinned:** Dockerfile with mutable base tags, unpinned package indexes, or a Dev Container with mutable Features.

Reproducibility should include repeated evaluation, deterministic seeds, fixed CPU/runtime settings where performance affects scoring, and an audit record of all hashes. An environment that installs successfully but evaluates a model with a different observation schema is not reproducible for competition purposes.

### Cross-platform support

- Native Python source is the most portable conceptual interface, but NumPy/PyTorch wheels are platform-specific and Python versions matter.
- A pure-Python wheel is portable only for the package portion; a wheelhouse needs separate platform artifacts.
- Docker standardizes Linux userspace but still requires a compatible host engine/VM and architecture image.
- PyInstaller requires target-specific builds and can expose libc/runtime issues.
- Dev Containers inherit Docker’s host/editor matrix.

For a college event, minimize the supported matrix. A well-tested Linux x86-64 target with an offline wheelhouse and optional Docker is operationally safer than claiming native support for every OS while testing only one. If broad access is required, provide a supported native path for Windows/macOS/Linux and make Docker optional rather than pretending Docker removes all platform friction.

### Offline use

**Best offline sequence:** distribute a release archive and wheelhouse; optionally distribute a Docker image tar. GitHub source archives work offline after download, but dependency retrieval does not. A package index is not an offline solution unless all artifacts are cached locally. A Dockerfile is not an offline solution unless base layers and package files are already present.

The release should be transferable by campus file share/USB and include a hash manifest. Test from a fresh machine with network disabled, not merely from a developer’s warm pip/Docker cache. Provide a small smoke test that fails with a clear message if the wrong release is installed.

### Trust and cheat resistance

Ranked from strongest to weakest for this event:

1. Model-only artifact; fixed server evaluator; private held-out tracks; no participant code execution.
2. Participant code in organizer-owned fixed sandbox with strict limits and no private-data access.
3. Participant-provided wheel/source/executable run by an organizer sandbox.
4. Participant-provided Docker image run by the organizer; more complex attack surface and image handling.
5. Local participant score or replay accepted as the result; not acceptable for a serious leaderboard.

Public source is not inherently bad: it improves fairness and learning when everyone sees the same environment. It only means that secrecy cannot be the defense. Anti-cheat should instead enforce that the server chooses the track/seed/scoring, the model cannot read private data, repeated deterministic runs agree, resources are bounded, and every submission is retained by hash.

### Setup friction

- Lowest for a trained programmer: public source + `venv` + wheelhouse.
- Lowest for a non-Python end user: executable, but the organizer pays the compatibility cost and students lose inspectability.
- Lowest for a reproducibility specialist: Docker/Dev Container, but Docker install and virtualization are real barriers.
- Highest educational value: source repository, because students can see and modify the environment.
- Best compromise: provide native Python as the required path and Docker/Dev Container as an opt-in path.

### Updates and versioning

Use immutable release artifacts and a compatibility policy:

- patch release: documentation/build fix that does not change the simulation or submission contract;
- minor/event release: public environment or schema change; new training and evaluation release;
- security rebuild: new evaluator image digest with a documented policy for whether prior scores are re-run;
- submission version: model artifact hash and manifest, never an overwritten filename.

Maintain a release table with source commit, package version, wheel hashes, wheelhouse hashes, Docker digest, evaluator version, public track manifest hash, and private track-set identifier. Reject submissions whose manifest claims a different event release unless the rules explicitly allow conversion.

### Licensing and institutional policy

Decide four separate rights questions:

1. **Starter code:** what license lets students clone, modify, submit, and publish forks?
2. **Third-party dependencies:** what notices/attributions must accompany source, wheels, images, and executables?
3. **Tracks/assets:** who owns and licenses geometry, textures, logos, and recorded replays?
4. **Participant artifacts:** does the organizer have permission to execute, retain, audit, display, and publish model files/results? Does a participant retain copyright and choose a model license?

Use SPDX identifiers and actual license files. PyPA’s licensing guidance is particularly useful for wheel/source distribution because the metadata and license files travel with the artifact. ONNX metadata can carry a model license, but event rules still need a submission license/usage grant. Avoid implying that a public GitHub repository or public leaderboard transfers participant IP.

## Concrete release plan

### Public release contents

```text
participant-release-2026.1/
  README.md
  LICENSE
  THIRD_PARTY_NOTICES/
  RELEASE.json
  SHA256SUMS
  participant-source.tar.gz
  racing_event-<version>-py3-none-any.whl
  requirements-offline-<platform>.txt
  wheelhouse/<platform>/*.whl
  examples/
  tracks/train/
  tools/validate_model.py
  tools/run_local_evaluation.py
  tests/
  Dockerfile
  docker-compose or .devcontainer/   # optional
```

`RELEASE.json` should state the Python version, package version, schema IDs, public track hashes, command entrypoints, supported platforms, model size limit, and model format. Include SHA-256 for every distributed artifact. A GitHub release can host the archive assets; a campus mirror is useful for offline/event-day availability.

### Model submission format

Recommended minimum:

```text
submission.zip
  model.onnx
  manifest.json
```

The manifest should contain:

- `event_release`;
- `model_format` and supported ONNX IR/opset;
- exact input/output names, dtypes, and shapes;
- model file SHA-256 and byte count;
- producer name/version;
- training seed(s), optional and informational unless rules score them;
- participant/team identifier supplied by the submission system, not trusted from the file;
- optional `model_license` and attribution; and
- optional human-readable description.

Reject extra files unless the rules deliberately support them. If recurrent/stateful policies are later permitted, make state tensors part of a new schema version instead of accepting arbitrary callback code.

### Pre-event tests

- Fresh Windows/macOS/Linux native setup from the release archive with network disabled.
- Fresh Docker setup from an image tar with network disabled.
- Public baseline trained/evaluated through the exact submission validator.
- Malformed ONNX, wrong shape, wrong dtype, oversized model, unsupported opset, custom op, external data, NaN/Inf, archive traversal, duplicate manifest, and timeout tests.
- Repeat the same model evaluation and compare result/trajectory hashes.
- Run the evaluator under CPU/memory/time limits and verify it cannot read a private-track path or reach the network.
- Confirm that the public repository, Git history, Docker layers, wheel metadata, logs, and release assets contain no private geometry, credentials, answer keys, or signing material.
- Test scoring queue recovery, worker death, duplicate submission, rerun, appeal, and leaderboard publication.

## Sources and date ledger

All links below were accessed on **2026-10-17**. Dates in parentheses are page/release dates where the source exposed them; otherwise the access date is the useful freshness marker.

### Python, packaging, and pip

- Python 3.11 `venv` documentation (page reports last updated 2026-10-01): https://docs.python.org/3.11/library/venv.html
- PyPA Packaging Python Projects tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/
- PyPA `pyproject.toml` specification and versioning history: https://packaging.python.org/en/latest/specifications/pyproject-toml/
- PyPA wheel/binary distribution specification: https://packaging.python.org/en/latest/specifications/binary-distribution-format/
- PyPA licensing examples and user scenarios / PEP 639 guidance: https://packaging.python.org/en/latest/guides/licensing-examples-and-user-scenarios/
- PyPA packaging/distributing projects guide (page marked “Outdated,” last reviewed 2023-12-14; retained for wheel/sdist concepts, with current PyPA specs preferred): https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/
- Pip repeatable installs: pinning, hashes, and wheelhouses: https://pip.pypa.io/en/stable/topics/repeatable-installs/
- Pip requirements file format and `--no-index`/`--find-links`/hash options: https://pip.pypa.io/en/stable/reference/requirements-file-format/

### GitHub

- GitHub repository model, public/private visibility, clone/fork/history: https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories
- GitHub source code archives, commit/tag stability caveats, and archive URLs: https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives
- GitHub repository visibility caveats: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility
- GitHub repository mirroring/duplication: https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository

### PyInstaller

- What PyInstaller bundles, target specificity, one-folder/one-file behavior, and bytecode/source caveats: https://pyinstaller.org/en/stable/operating-mode.html
- PyInstaller command/build options: https://pyinstaller.org/en/stable/usage.html
- PyInstaller platform requirements and per-platform bootloader/build notes: https://pyinstaller.org/en/stable/requirements.html
- PyInstaller troubleshooting, hidden imports, `noexec`, and cross-platform notes: https://pyinstaller.org/en/stable/when-things-go-wrong.html

### Docker and Dev Containers

- Docker build best practices, mutable tags versus digest pinning, multi-stage builds, non-root users, and CI: https://docs.docker.com/build/building/best-practices/
- Docker Engine security: namespaces, cgroups, daemon/host-mount risk, capabilities, rootless mode, and trust boundaries: https://docs.docker.com/engine/security/
- Docker resource constraints: CPU and memory limits: https://docs.docker.com/engine/containers/resource_constraints/
- Docker Desktop installation/platform terms: https://docs.docker.com/desktop/setup/install/
- Docker image save for offline transport: https://docs.docker.com/reference/cli/docker/image/save/
- Docker image load from tar/compressed archive: https://docs.docker.com/reference/cli/docker/image/load/
- Docker reproducible-build timestamps with `SOURCE_DATE_EPOCH` (GitHub Actions docs): https://docs.docker.com/build/ci/github-actions/reproducible-builds/
- Development Container Specification: https://containers.dev/implementors/spec/
- Dev Container Dockerfile/Compose usage (dated 2022-12-16): https://containers.dev/guide/dockerfile

### Model format

- ONNX IR specification: versioned IR/opsets, model checker, inference graph semantics, metadata including `model_license`, custom/external-data concepts: https://github.com/onnx/onnx/blob/main/docs/IR.md
- ONNX Runtime documentation index/tutorials: https://onnxruntime.ai/docs/tutorials/
- PyTorch serialization semantics, including pickle defaults and `weights_only` limitations (created/updated 2026-05-08): https://docs.pytorch.org/docs/2.14/notes/serialization.html
- PyTorch `torch.load` API warning not to load untrusted data: https://docs.pytorch.org/docs/2.14/generated/torch.load.html

### Competition practice

- Kaggle official CLI repository (Apache-2.0; competition/model upload/version workflows): https://github.com/Kaggle/kaggle-cli
- Kaggle competition commands, submission files/code competitions/status: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competitions.md
- Kaggle competition creation, staged pages, public data versus private rerun data, and frozen solution: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competition_creation.md
- Kaggle CLI tutorials, competition/code/model/model-version submission patterns: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/tutorials.md
- Codabench participant result/code zip submissions: https://docs.codabench.org/latest/Participants/User_Participating-in-a-Competition/
- Codabench organizer-controlled competition Docker image: https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Competition-docker-image/
- Codabench submission process and compute workers: https://docs.codabench.org/latest/Developers_and_Administrators/Submission-Process-Overview/
- Codabench submission container layout and reference-data boundary: https://docs.codabench.org/latest/Developers_and_Administrators/Submission-Docker-Container-Layout/
- Codabench participant routing to compute queues: https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Participant-Routing/
- Codabench source repository, Apache-2.0: https://github.com/codalab/codabench
- AWS DeepRacer on AWS implementation guide (publication date September 2026): https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/
- AWS DeepRacer on AWS public implementation repository and architecture/validation workflow: https://github.com/aws-solutions/deepracer-on-aws

## Bottom line

For this event, choose **public source + versioned package/offline bundle + optional Docker/Dev Container + model-only ONNX submission + private server-side evaluation**. It gives students inspectable Python and a low-friction local path, gives organizers an immutable release/audit trail, works without network access after distribution, and avoids making arbitrary participant programs the default production workload. Treat standalone executables and participant-supplied containers as optional conveniences or separate code-competition mechanisms, not as the core fairness boundary.
