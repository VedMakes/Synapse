# SYNAPSE participant delivery options and racing maps

**Research date:** 2026-10-17

**Question covered:** how to prepare and deliver training maps for a racing RL
event while preserving held-out-track secrecy and fair difficulty. This note
covers original/procedural maps, real-circuit-inspired geometry, file formats,
versioning and hashes, public/private delivery, train/test splitting, map
validation, licensing/IP, and map-count/variety experiments.

**Researcher-guidance access note:** the exact requested file
`/home/redacted/.claude/skills/synced/c7f094ac-bed3-482b-b469-18116bcbb757_6f09fcc4-c667-4fc5-8e93-66954d59222e/deep-research/references/researcher.md`
was not present in the mounted container. I attempted to read it before the
research step. The available deep-research skill instructions were used, and
all web sources below were checked on the date above.

## Short recommendation

Use **original, procedurally authored 2D geometry** as the event’s authoritative
map source. Give participants a public, immutable package containing the
environment, schema, validator, builder examples, and **three to five** public
training maps. Keep **two or three** final maps private and load them only on
the tournament evaluator. Use real circuits only as broad design references,
or use a properly licensed geographic source in a documented transformation
pipeline. Do not trace, ship, or publicly demonstrate an exact recognizable
circuit without a circuit/IP-owner clearance review.

The current repository is already close to this model:

- `tools/track_builder.py` creates three explicitly original synthetic loops.
- `racing_event/track.py` defines a JSON geometry version (`1.0`), canonical
  content hashing, finite-value and schema checks, closed-loop geometry checks,
  nested boundaries, runoff, and barriers.
- `tracks/train/*.json` is participant-visible training content. The local
  README and handoff state that held-out tracks are a planning target and must
  not be included in participant-facing content.
- `docs/replay_contract.md` correctly warns that private held-out geometry must
  not be exported to public previews and that checksums are integrity checks,
  not anti-cheat authentication.

This note does not change application code.

## Evidence and implications

### Procedural/original synthetic maps

**Evidence.** Procgen’s benchmark paper introduced 16 procedurally generated
environments and reports that diverse environment distributions are essential
for both sample efficiency and generalization. Its central evidence is not a
race-specific map-count prescription, but it supports varying the underlying
track distribution instead of training on one fixed layout.

- Karl Cobbe et al., *Leveraging Procedural Generation to Benchmark
  Reinforcement Learning*, submitted 2019-12-03, revised 2020-07-26, accessed
  2026-10-17:
  [arXiv abstract](https://arxiv.org/abs/1912.01588)

**Recommendation.** Generate tracks from a pinned, reviewable family of
centerline primitives or splines. Store the generated output, not only the
seed. A generator should expose a small set of meaningful parameters:

- lap length or common scale;
- corner-radius and curvature limits;
- straight-section length distribution;
- corner direction sequence, including hairpins, chicanes, sweepers, and
  linked turns;
- track width and runoff policy;
- start-line position; and
- a generator version and private/public seed record.

The generator must reject invalid outputs before they become training or test
assets. A public generator is useful for participant experimentation, but a
private held-out seed is not secret if the participant can infer it from a
published map preview, parameter tuple, generator commit, or deterministic
seed search. For the final maps, keep the seed and exact generator invocation
private, or author the output privately and publish only the post-event
artifact.

**Advantages.** Original procedural maps give the organizers control over
difficulty, variety, scale, copyright provenance, and reproducibility. They
also make it possible to create a matched train/test distribution without
shipping a famous circuit’s recognizable outline.

**Risks.** A weak generator can produce many cosmetic variants of the same
track. Agents then memorize a latent template rather than learn transferable
racing behavior. Excessive randomization can also create maps that are
technically valid but unreasonable for the fixed vehicle. The generator must
therefore be constrained by geometry and behavioral difficulty checks, not just
randomness.

### Real-circuit-inspired geometry

There are three different sourcing choices and they should not be conflated:

1. **Generic racing inspiration:** use public observations and racing design
   knowledge to specify broad characteristics such as a long straight, a
   decreasing-radius corner, or a technical sector. Author a new centerline,
   name, boundary, and visual identity. This is the recommended use.
2. **Licensed geographic input:** use an input whose licence permits the
   intended extraction, adaptation, and public delivery. Record the source,
   version/date, exact files, licence text, attribution, and transformation.
   OpenStreetMap is possible, but its ODbL obligations must be handled; it is
   not a blanket clearance for an exact named circuit.
3. **Exact recognizable circuit:** copy or closely redraw a circuit layout,
   use its name, or reproduce its signs/assets. Treat this as a separate
   licensing/IP project. Do not assume that renaming the track, making the
   event educational, or charging no money resolves the issue.

**Authoritative IP evidence.** WIPO lists maps, technical drawings, databases,
architecture, and computer programs among commonly protected works while
distinguishing expression from ideas, procedures, methods, and mathematical
concepts. The U.S. Copyright Office gives the same facts-versus-expression
boundary. Formula 1’s current guidelines specifically direct circuit-outline
licensing enquiries to the circuit IP owner and separately restrict Formula 1
marks, official assets, timing data, and third-party racing simulators.

- WIPO, *Copyright*, publication date not stated, accessed 2026-10-17:
  [https://www.wipo.int/copyright/en/](https://www.wipo.int/copyright/en/)
- WIPO, *Copyright FAQ*, publication date not stated, accessed 2026-10-17:
  [https://www.wipo.int/en/web/copyright/faq-copyright](https://www.wipo.int/en/web/copyright/faq-copyright)
- U.S. Copyright Office, *Copyright in General FAQ*, publication date not
  stated, accessed 2026-10-17:
  [https://www.copyright.gov/help/faq/faq-general.html](https://www.copyright.gov/help/faq/faq-general.html)
- Formula 1, *Guidelines*, current page accessed 2026-10-17:
  [https://www.formula1.com/en/information/guidelines.4EOKE9RRqevL4niTK9kWyt](https://www.formula1.com/en/information/guidelines.4EOKE9RRqevL4niTK9kWyt)
- Formula 1, *Legal Notices*, publication date not stated, accessed
  2026-10-17:
  [https://www.formula1.com/en/information/legal-notices.7egvZU48hzrypubGBNcQKt](https://www.formula1.com/en/information/legal-notices.7egvZU48hzrypubGBNcQKt)

**Recommendation.** Use neutral track IDs and fictional series branding. If a
map is described as inspired by a real circuit, change the layout materially:
do not preserve exact coordinates, a distinctive turn sequence, official
signage, landmarks, circuit name, sponsor marks, or game-derived assets. Keep
a source ledger even when the final geometry is independently authored.

## Geometry file formats

### Recommended event format: the repository’s canonical JSON

Keep one small, self-contained geometry document per track. The current local
schema contains:

```text
version
centerline, s, L, heading, curvature
half_width_left, half_width_right
left_boundary, right_boundary
runoff_width, barrier_left, barrier_right
start_line_s
meta: id, display_name, source_note, geometry_version, content_hash, metrics
```

The centerline and all rings are implicit closed loops: the final point is
unique and the closing segment returns to the first point. This is a good
authoritative representation for the current Python physics and Unity
renderer because both consume the same explicit boundaries. It is human
reviewable, easy to diff, has no hidden binary decoder, and carries the
features needed by projection, raycasts, car-footprint tests, and rendering.

Use JSON for the **published and hashed canonical artifact**. NPZ or another
binary cache may be generated locally for speed, but it must be derived from
the canonical JSON and checked against its hash. Do not let a renderer or a
trainer silently choose a different resampling, closure convention, unit, or
coordinate transform.

### Interoperability formats and their appropriate role

| Format/reference | Evidence | Good use | Limitation for this event |
|---|---|---|---|
| **ASAM OpenDRIVE (`.xodr`, `.xodrz`)** | ASAM describes current version 1.9.0, released 2026-05-19, as an XML standard for road networks, lanes, objects, signals, elevation, and simulator exchange. It supports both synthetic and real data. [ASAM OpenDRIVE](https://www.asam.net/standards/detail/opendrive/), accessed 2026-10-17. | Import/export boundary when interoperability with road simulators is required; preserve a source-level reference line and lane metadata. | Overkill for one closed racing loop. XML ordering, defaults, geometry primitives, lane semantics, and conversion can introduce hash/determinism problems. Treat it as an interchange or authoring source, then normalize into event JSON. |
| **F1TENTH YAML + grayscale image + centerline CSV** | F1TENTH documents a custom map as a YAML metadata file plus a single-channel black/white image; `resolution` is metres/pixel and `origin` uses the lower-left world convention. Its beta random generator also emits map files and centerline CSV, with a reproducible seed. [Custom map](https://f1tenth-gym.readthedocs.io/en/stable/customized_usage.html#custom-map), accessed 2026-10-17. | Useful for occupancy/raycast experiments, a rasterized compatibility export, or comparing against an authoritative racing RL environment. | Raster resolution and image-origin conventions can alter boundaries. The image alone does not carry the current event’s variable widths, runoff, barriers, curvature, start line, or content provenance. |
| **OpenDRIVE-derived simulator maps** | CARLA says its map road definition is based on OpenDRIVE and exposes waypoints, lanes, junctions, and map conversion. [CARLA maps](https://carla.readthedocs.io/en/latest/core_map/), accessed 2026-10-17. | Reference for validating that a road-network format can drive multiple tools and for future road-network extensions. | CARLA’s 3D town/map package is much broader than the current 2D time-trial task. Do not use CARLA or third-party map assets as event geometry unless their asset terms and provenance are cleared. |
| **CommonRoad XML** | CommonRoad describes composable benchmarks identified by unique IDs, combining vehicle model, cost function, and scenario; scenarios are partly real and partly hand-crafted. The `commonroad-io` package reads/writes the scenario format. [CommonRoad README](https://github.com/CommonRoad/commonroad-io/blob/master/README.md), accessed 2026-10-17. | Useful research benchmark reference for scenario identity, reproducibility, and separating scenario data from vehicle/cost definitions. | It models road networks, obstacles, trajectories, goals, and planning problems, not the minimal single-loop racing schema. Adopting it would add complexity without solving held-out secrecy. |
| **Render meshes (`glTF`, FBX, Unity assets)** | These are common interchange/rendering choices, but they are not the authoritative sources in the local Python/Unity contract. | Derived presentation asset after Python geometry is frozen. | Meshes can omit centerline semantics, produce boundary drift, carry separate third-party rights, and accidentally leak private map shape through screenshots or prefabs. |

**Format recommendation.** Keep the current JSON as the event contract. If an
OpenDRIVE or F1TENTH import is ever accepted, define a one-way conversion:
source -> normalized centerline -> uniform arclength samples -> derived
boundaries/barriers -> validation -> canonical JSON. Hash the normalized JSON,
not the source file’s incidental XML/PNG bytes. Keep the source licence and
source hash in the provenance ledger.

## Versioning, content hashes, and manifests

### Two identities per map

Record both of these values:

1. **Canonical content hash:** SHA-256 of canonical JSON with
   `meta.content_hash` excluded while calculating the digest. This is the
   identity already implemented by `TrackGeometry.content_hash_for_data`.
2. **Delivered-file hash:** SHA-256 of the exact bytes in the release or
   private evaluator package. This detects packaging or newline/encoding
   changes even when the parsed content is equivalent.

NIST FIPS 180-4 describes secure hash digests as a way to detect whether a
message changed since its digest was generated. It supports using SHA-256 for
integrity, but a plain hash is not a signature or proof of who supplied a map.

- NIST, *Secure Hash Standard*, published 2015-08-04, updated 2024-07-25,
  accessed 2026-10-17:
  [https://www.nist.gov/publications/secure-hash-standard](https://www.nist.gov/publications/secure-hash-standard)

Use a manifest with at least:

```text
manifest_version
event_release_id
geometry_version
map_id
split: train | dev | final_private
canonical_content_hash
delivered_file_sha256
units and coordinate convention
generator/build version
validator version and result
source/license record ID
difficulty summary (public for train/dev, private for final)
```

Do not put the final map’s clear-name, seed, parameter vector, preview, SVG,
Unity prefab, or derived centerline in the public manifest. A final-map hash can
be kept in the private evaluator manifest and checked at runtime. If the
organizers publish a commitment hash before the event, understand that a
participant who already has candidate geometry can compare hashes; this is a
commitment to a chosen file, not secrecy.

Pin separately:

- environment code hash;
- simulation configuration hash;
- observation/action schema versions;
- public training-map manifest hash;
- private final-map manifest hash, held only by the evaluator; and
- evaluator release/tag and dependency lock.

The local replay contract already follows the right principle: its SHA-256
checks detect accidental changes, but are explicitly not anti-cheat
authentication. Tournament authenticity needs access controls, a controlled
evaluator, and an audit log in addition to hashes.

## Public versus private participant delivery

### Public participant package

Publish an immutable, tagged release containing:

- the fixed Python environment and trainer boundary;
- the public training geometry and public map manifest;
- the exact schema/units/closure rules;
- the validator and a command that validates every published map;
- a small original-map builder or examples, if participants need map
  authoring practice;
- the event configuration and dependency lock;
- licence texts and attribution notices for every shipped external input; and
- sample replays containing only public training geometry.

The package should be reproducible from a clean machine and should not need a
network at training time. A Git tag/release is useful for a public artifact:
GitHub documents releases as deployable iterations based on tags and supports
release assets. Publish a frozen release only after the train maps, code, and
manifest have been checked together.

- GitHub, *Licensing a repository*, accessed 2026-10-17:
  [https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- GitHub, *About releases*, accessed 2026-10-17:
  [https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- GitHub, *Managing releases in a repository*, accessed 2026-10-17:
  [https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)

GitHub’s licensing guidance says that without a licence, default copyright law
applies and others generally may not reproduce, distribute, or make derivative
works from the code. A repository licence also does not automatically clear
third-party data, art, trademarks, or map assets. Every source file or asset
must be checked on its own terms.

### Private final-map package

Keep final tracks outside any participant-visible repository and outside its
history, issues, pull requests, Actions artifacts, release assets, container
images, public object storage, and public replay previews. The evaluator should
receive a private package through an organizer-controlled path immediately
before the evaluation window, then verify the private manifest and map hashes
before running policies.

Recommended separation:

```text
public release
  env code, config, schema, validator
  tracks/train/*.json
  public manifest and notices

private evaluator bundle
  exact held-out JSON files
  private manifest and difficulty report
  evaluation runner and private configuration
  access/audit record
```

VECTOR or a renderer operator should receive only the private geometry needed
to create the show assets, under the same access boundary. Do not send a
private map to a public Unity package, public screenshot, public replay, or
participant troubleshooting log. If a private track must be previewed for
production, use a crop or abstract difficulty card that does not reveal the
route topology, and keep the full geometry private.

GitHub’s public-repository guidance says public repositories are visible and
forkable; changing visibility later does not remove existing forks or local
copies. That is why deleting a held-out file in a later commit is not a secrecy
repair. GitHub also documents that Git LFS uses pointer files in the repository
while the large object is stored separately; LFS is a storage mechanism, not an
access-control or secrecy solution. Keep private maps out of the participant
repository rather than relying on LFS or an ignored file.

- GitHub, *Licensing a repository* (public repositories and forks), accessed
  2026-10-17:
  [same page](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- GitHub, *About Git Large File Storage*, accessed 2026-10-17:
  [https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- GitHub, *Duplicating a repository* (mirror clones include branches/tags and
  can include LFS objects), accessed 2026-10-17:
  [https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository)

### Secrecy red-team before release

Have a reviewer with only participant access inspect:

- all Git branches, tags, deleted-file history, and release archives;
- GitHub issues, discussions, pull requests, CI logs, caches, and artifacts;
- Docker layers and package contents;
- Unity scenes, prefabs, baked meshes, thumbnails, screenshots, and video;
- docs, notebooks, debug output, exception messages, and map IDs;
- public code for a deterministic generator and its possible seed space; and
- replay export paths and browser bundles.

The reviewer should not be able to reconstruct the held-out route from any
of these. Run this check after the final release is tagged, not only on the
working tree.

## Train/test split and fair difficulty

### Split by route family, not by points

Keep every complete route and all of its derived variants in one split. Do not
put a smoothed, mirrored, rescaled, or seed-neighbour version of a final map in
training. If a procedural generator has correlated seeds, cluster maps by
geometry signature first and split clusters, not random individual files.

The observation contract should remain track-relative and omit absolute
position, absolute heading, track ID, and absolute lap progress. This prevents
an otherwise artificial route identifier from turning the task into map
memorization. Random training spawns are useful for section coverage, but
evaluation should use the fixed, documented standing start. The racing paper by
Czechmanowski et al. is a relevant precedent: it trained 400 parallel
instances and reset out-of-bounds rollouts to random track positions to improve
full-track exploration. That is evidence for a training-coverage mechanism,
not evidence that random respawns alone guarantee unseen-track generalization.

- Czechmanowski et al., *On learning racing policies with reinforcement
  learning*, arXiv v1 2025-04-03, accessed 2026-10-17:
  [HTML](https://arxiv.org/html/2504.02420v1)
- Fuchs et al., *Super-Human Performance in Gran Turismo Sport Using Deep
  Reinforcement Learning*, arXiv v2 2021-05-09, accessed 2026-10-17:
  [HTML](https://arxiv.org/html/2008.07971v2)

Fuchs et al. report that a policy transferred to a different track handled
straight and slight-curve sections but failed to extrapolate to some unseen
curve shapes. This supports evaluating route-family diversity, not merely
counting files. The paper also reports that fixed action-change limits were
used to make an agent comparison fair, which is a useful reminder that map
fairness cannot compensate for a changing policy/evaluation contract.

### Difficulty dimensions to balance

Compute a summary for every map, with the summary itself private for final
tracks until the event:

- lap length and expected lap time;
- minimum and percentile corner radius;
- signed curvature distribution and curvature variation;
- number and severity of hairpins, chicanes, linked turns, and sweepers;
- longest straight and proportion of low-curvature distance;
- width and width variation relative to the fixed car width;
- runoff and barrier distances;
- total left/right turn balance and reversal frequency;
- start-line context and first-corner severity; and
- reference-policy completion, lap time, off-track distance, and crash rate.

Match **distributions**, not just means. For example, a training set with one
hairpin and a final set with three hairpins is not fair even if average lap
length and minimum radius match. Conversely, exact geometric matching can make
the private maps easy to infer. Publish only coarse training/dev summaries and
keep the final summary private until the split is frozen.

### Difficulty parity protocol

Before freezing the event:

1. Select a reference policy that was not tuned on the final maps. A simple
   deterministic controller or the fixed local PPO baseline is sufficient for
   the parity test.
2. Run the same vehicle, physics, action limits, start rule, seed policy, and
   step limit on every train, development, and final candidate map.
3. Repeat over several seeds or fixed spawn points. Record completion rate,
   lap time, crash rate, off-track distance, stuck/timeout rate, and section
   coverage.
4. Compare standardized metrics and route-feature distributions. Pre-register
   an acceptable tolerance before looking at final-track results. A reasonable
   initial engineering gate is a roughly 10% median reference-performance
   difference and no materially different failure-rate tail, then adjust only
   from pilot evidence. This is a project threshold, not a universal research
   law.
5. Reject or repair a final map that is an outlier, or add a compensating map
   so the final aggregate has the same difficulty mixture as training.
6. Freeze the private maps and manifest before participant training begins.

CARLA is a useful competition precedent: its documentation labels Town08 and
Town09 as secret unseen towns for its leaderboard challenge. The useful design
lesson is the separation of public development maps from secret evaluation
maps. It does not establish a required count or a legal model for this event.

- CARLA, *Maps and navigation*, current documentation accessed 2026-10-17:
  [https://carla.readthedocs.io/en/latest/core_map/](https://carla.readthedocs.io/en/latest/core_map/)

## Map validation gate

Validation should have separate geometry, physics, determinism, difficulty, and
delivery stages. A map is not ready because it renders plausibly.

### Geometry and numeric checks

- Parse only the declared schema/version; reject unknown required-shape fields,
  NaN, infinity, duplicate points, zero-length segments, and malformed closure.
- Require a simple non-self-intersecting centerline. The current 2D projection
  and raycast model should reject crossings and overpasses because two different
  route segments can project to the same local area.
- Require approximately uniform arclength sampling and verify that `s` and `L`
  agree with the measured closed-loop length.
- Verify positive widths, a fixed car-to-track-width ratio, and physically
  drivable minimum radius.
- Verify left/right boundaries and inner/outer barriers are consistently
  oriented, nested, non-crossing, and separated by the promised runoff width.
- Verify the full rectangular car footprint can start at the evaluation line
  without immediate contact and that it can traverse the map without a
  topological dead end.
- Check curvature and heading continuity after resampling. Flag sharp sample
  artifacts that are not intentional corners.
- Check raycasts, projection, width lookup, edge status, and one complete lap
  at representative positions, including near the closure seam.

These checks match the local `TrackGeometry.validate()` contract and the
existing tests. They should run in CI and in the evaluator, with the evaluator
using the exact pinned validator version.

### Physics and behavioral checks

- Run a scripted controller and a conservative reference controller.
- Require deterministic repeated trajectories for the same map, config, seed,
  and action stream; compare trajectory digests.
- Check that a one-byte map change changes the canonical content hash and is
  rejected by a stale manifest.
- Check that fixed-start evaluation has no random respawn, no hidden second
  chance, and no private-map replay export.
- Measure completion and failure outcomes under a modest set of starts. A
  route that can be validated geometrically but has no feasible lap for the
  pinned car is not a fair map.
- Compare Python state and any Unity/renderer mesh round trip against the same
  boundary coordinates. The renderer may display a map but must not recompute
  collision, progress, rewards, or timing.

ASAM’s Quality Checker framework is a useful format-validation precedent. It is
standard-agnostic, supports checker bundles, and has official OpenDRIVE and
OpenSCENARIO checker libraries. It does not validate the event’s bespoke racing
semantics, so the event still needs its own checks for car footprint, curvature,
runoff, barriers, and difficulty.

- ASAM, *Quality Checker Framework*, current page accessed 2026-10-17:
  [https://www.asam.net/standards/asam-quality-checker/](https://www.asam.net/standards/asam-quality-checker/)

## Licensing and IP checklist

### OpenStreetMap and ODbL

OpenStreetMap’s official copyright page says OSM data is licensed under the
Open Data Commons Open Database License (ODbL). It summarizes permission to
copy, distribute, transmit, and adapt with credit, and says altered or
built-upon data may need to be distributed under the same licence. It requires
an attribution notice and a clear link/notice to the ODbL.

The ODbL legal text is more precise:

- it grants worldwide, royalty-free use of database rights subject to its
  conditions;
- a publicly conveyed database or derivative database needs the licence and
  notices;
- a public produced work needs a notice that its content was obtained from the
  database and is available under ODbL;
- a derivative database can trigger share-alike and machine-readable access to
  the derivative or alteration file; and
- ODbL does not cover independent copyright or other rights in individual
  contents, and it does not cover trademarks associated with the database.

Therefore, if OSM is used, keep the raw extract, extract date, query, source
URL, transformation script/version, output hash, and attribution text in the
ledger. Include a visible credit such as “Contains information from
OpenStreetMap, which is made available under the Open Database License” with
links to the OSM copyright page and ODbL text. Determine whether the shipped
artifact is a produced work or a derivative database with qualified legal help;
do not assume a rasterized or simplified track automatically avoids all
obligations. OSM also warns against importing data from copyrighted sources
without permission and does not provide a free third-party map API or tile
service.

- OpenStreetMap Foundation, *Copyright and License*, publication date not
  stated, accessed 2026-10-17:
  [https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)
- Open Data Commons, *ODbL v1.0 legal code*, publication date not stated,
  accessed 2026-10-17:
  [https://opendatacommons.org/licenses/odbl/1-0/](https://opendatacommons.org/licenses/odbl/1-0/)

### GitHub repositories and licences

GitHub’s official licensing guide says a repository with no licence remains
under default copyright law. A public repository may be visible and forked,
but visibility/forking does not mean the data or assets are licensed for event
use. A detected repository licence should be read with the exact file paths,
not inferred from a README badge alone. A code licence may not cover embedded
maps, textures, names, trademarks, datasets, or generated assets.

Use unlicensed racing repositories as read-only references. For any external
map or tool that survives the provenance review, record the exact repository,
commit/tag, file path, licence snapshot, author/rights holder, permissions for
modification and public redistribution, attribution/share-alike terms, and
third-party-content exclusions. Preserve required notices in the public
release. If the licence or ownership cannot be verified, replace the asset
with an original one.

### Practical event policy

Lowest-risk policy:

- fictional event, track names, signs, sponsors, car designs, and visual
  identity;
- original centerlines, boundaries, runoff, barriers, and render meshes;
- no ripped game files, game screenshots, official maps/CAD, telemetry,
  timing-data dumps, logos, typefaces, audio, or branded circuit assets;
- public source/licence ledger and notices for all external inputs; and
- written contributor/college agreements for student-created geometry, art,
  code, audio, and documentation.

If the required feature is “the player must recognize an exact real circuit,”
pause and obtain written permission from the circuit/IP owner and any relevant
series/brand rights holders before public demonstration or distribution. This
note is engineering guidance, not jurisdiction-specific legal advice.

## Recommended map counts and variety experiments

There is no authoritative universal number of tracks for a racing RL event.
The existing handoff hypothesis of three to five training tracks and two to
three held-out tracks is a reasonable starting range. Use a pilot to select the
final number rather than deciding by intuition.

### Launch candidate

- **Training:** 5 maps if the fixed budget is large enough; otherwise 3 maps.
  Include at least one high-speed/sweeper map, one technical/hairpin map, one
  chicane or linked-turn map, and one mixed map. The fifth can be a moderate
  variant from a distinct route family, not a cosmetic duplicate.
- **Development holdout:** 1 or 2 maps known only to organizers and used for
  internal selection. Do not use the final maps to tune the generator,
  reward, or budget.
- **Final private holdout:** 2 or 3 maps, with the same broad feature and
  reference-difficulty mixture as training. Keep at least one final map from a
  route family not represented by a near-duplicate public map.
- **Sampling:** sample public training maps uniformly by map ID, then sample
  starts uniformly or by the documented spawn mixture. Do not let a long map
  dominate simply because it generates more steps unless that weighting is a
  deliberate, published rule.

### Experiments to run before freezing the event

Use the same environment version, reward, policy architecture, total simulated
steps, optimizer settings, evaluation protocol, and at least five random seeds
for each row. Keep one final private split untouched while choosing settings.

| Experiment | Conditions | Primary evidence | Decision |
|---|---|---|---|
| **E-map-count** | Train on `N ∈ {2, 3, 5, 8}` maps, with equal total steps. | Training completion, public-dev completion, final-private completion, normalized lap time, crash/off-track rate, seed variance. | Choose the smallest `N` whose final generalization gap is acceptable without making the fixed budget too dilute. The handoff already records this exact comparison as E3. |
| **E-variety** | Same count and budget: single route family, balanced hand-authored families, and constrained procedural samples. | Performance by corner type, not only aggregate lap time; failure on rare hairpins/chicanes; map-feature coverage. | Prefer the set that improves broad corner coverage without creating an unfair final difficulty tail. |
| **E-procedural-vs-inspired** | Original procedural set versus independently authored real-flavor set, with matched geometry statistics. | Generalization, human interpretability, validation rejects, provenance cost, and licence burden. | Use procedural maps for the release unless the inspired set has a clear, measured educational/show benefit and clean provenance. |
| **E-difficulty-parity** | Candidate train/final sets evaluated by the same reference controller over repeated starts. | Distribution of normalized reference lap times, completion, crashes, off-track distance, and curvature/width statistics. | Reject hidden easy/hard tracks; freeze a balanced private set. This is the handoff’s E9. |
| **E-secrecy-red-team** | Participant-only clone/release after each candidate packaging method. | Can a reviewer recover route shape from history, assets, logs, seeds, hashes, or previews? | Ship only the package that passes; deleting a leaked file in a later commit is insufficient. |
| **E-split-family** | Random file split versus cluster-by-geometry-family split. | Nearest-neighbour geometry similarity and zero-shot performance. | Use family-aware splitting whenever correlated procedural variants exist. |

### Metrics and reporting

Report per-map and aggregate results. At minimum:

- completion rate and DNF reason;
- fixed-step lap time for completed laps;
- crash and barrier-contact rate;
- off-track distance/time;
- mean and percentile speed;
- action smoothness/saturation;
- coverage by arclength section;
- reference-policy difficulty; and
- median, interquartile range, and worst-case results across seeds.

Do not select map counts using only training reward. The racing literature
shows that reward and physical outcomes can diverge, and the Fuchs paper’s
unseen-track failure is a direct warning that strong single-track performance
does not establish route generalization.

## Delivery checklist

### Before public release

- [ ] Freeze the geometry schema, units, closure convention, and validator.
- [ ] Generate/author public maps and run geometry plus reference-policy checks.
- [ ] Cluster related variants and create train/dev/final split by route family.
- [ ] Run map-count, variety, and difficulty-parity experiments.
- [ ] Create canonical and delivered-byte hashes and sign/tag the public release
      through the organizer’s controlled release process.
- [ ] Search public branches, tags, history, issues, CI logs, Docker layers,
      Unity assets, screenshots, replays, and archives for final-map leakage.
- [ ] Publish licences, attributions, source ledger, and contributor ownership
      records for content that is actually shipped.

### Before tournament evaluation

- [ ] Copy the private evaluator bundle through an organizer-controlled path.
- [ ] Verify private map manifest, file hashes, geometry version, config hash,
      code hash, and validator version.
- [ ] Run deterministic smoke laps and compare known trajectory digests.
- [ ] Confirm no private map is written to participant output, public replay,
      browser bundle, Unity package, or evaluation logs.
- [ ] Evaluate every submission with the same shared hardware, fixed start,
      fixed action/evaluation contract, and published DNF/ranking rules.
- [ ] Retain a private audit record of evaluator release, map hashes, policy
      hash, seed, and outcome.

## Source list with dates

The following sources were prioritized for authority and direct relevance.
All were accessed on **2026-10-17** unless another date is stated above.

1. OpenStreetMap Foundation, *Copyright and License*:
   [https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright)
2. Open Data Commons, *Open Database License v1.0*:
   [https://opendatacommons.org/licenses/odbl/1-0/](https://opendatacommons.org/licenses/odbl/1-0/)
3. GitHub Docs, *Licensing a repository*:
   [https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
4. GitHub Docs, *About releases*:
   [https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
5. GitHub Docs, *About Git Large File Storage*:
   [https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
6. NIST, *Secure Hash Standard*, FIPS 180-4:
   [https://www.nist.gov/publications/secure-hash-standard](https://www.nist.gov/publications/secure-hash-standard)
7. ASAM, *OpenDRIVE*:
   [https://www.asam.net/standards/detail/opendrive/](https://www.asam.net/standards/detail/opendrive/)
8. ASAM, *Quality Checker Framework*:
   [https://www.asam.net/standards/asam-quality-checker/](https://www.asam.net/standards/asam-quality-checker/)
9. F1TENTH Gym, *Custom map*:
   [https://f1tenth-gym.readthedocs.io/en/stable/customized_usage.html#custom-map](https://f1tenth-gym.readthedocs.io/en/stable/customized_usage.html#custom-map)
10. O’Kelly et al., *F1TENTH*, PMLR 123:77-89, 2020:
    [https://proceedings.mlr.press/v123/o-kelly20a.html](https://proceedings.mlr.press/v123/o-kelly20a.html)
11. Cobbe et al., *Leveraging Procedural Generation to Benchmark
    Reinforcement Learning*, arXiv 1912.01588, 2019/2020:
    [https://arxiv.org/abs/1912.01588](https://arxiv.org/abs/1912.01588)
12. Czechmanowski et al., *On learning racing policies with reinforcement
    learning*, arXiv 2504.02420v1, 2025-04-03:
    [https://arxiv.org/html/2504.02420v1](https://arxiv.org/html/2504.02420v1)
13. Fuchs et al., *Super-Human Performance in Gran Turismo Sport Using Deep
    Reinforcement Learning*, arXiv 2008.07971v2, 2021-05-09:
    [https://arxiv.org/html/2008.07971v2](https://arxiv.org/html/2008.07971v2)
14. CARLA, *Maps and navigation*:
    [https://carla.readthedocs.io/en/latest/core_map/](https://carla.readthedocs.io/en/latest/core_map/)
15. CommonRoad, `commonroad-io` README:
    [https://github.com/CommonRoad/commonroad-io/blob/master/README.md](https://github.com/CommonRoad/commonroad-io/blob/master/README.md)
16. WIPO, *Copyright*:
    [https://www.wipo.int/copyright/en/](https://www.wipo.int/copyright/en/)
17. U.S. Copyright Office, *Copyright in General FAQ*:
    [https://www.copyright.gov/help/faq/faq-general.html](https://www.copyright.gov/help/faq/faq-general.html)
18. Formula 1, *Guidelines*:
    [https://www.formula1.com/en/information/guidelines.4EOKE9RRqevL4niTK9kWyt](https://www.formula1.com/en/information/guidelines.4EOKE9RRqevL4niTK9kWyt)
