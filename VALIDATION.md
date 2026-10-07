# Implementation verification

## Simplified tank management and display controls — 7 October 2026

- **92 regression tests passed**. New coverage includes confirmed stock removal,
  changed-population rejection, concurrent retries, preserved history and tank
  codes, daily eligibility after removal, rejection of new operations on removed
  tanks, and worker/validation cleanup without interrupting another tank.
- Removal records remaining population as `tank_removed`, separately from deaths
  and dispersals. Tests verify existing mortality and historical population remain
  correct. Dispersals now match multiple recent camera losses chronologically,
  including partial matches and quantities exceeding the already observed loss.
- Image upload and reprocessing return unannotated previews and separate detection
  metadata. Raw-preview regressions verify annotations are absent. The UI no
  longer silently falls back to an image with baked-in overlays.
- Desktop/mobile browser checks passed for individual video and photo overlays,
  Clean video, trail controls, private preferences, camera accuracy setup, tank
  opening by mouse/keyboard/touch, simplified dispersal fields and outgoing-transfer
  submissions, and deletion with a concurrent population change. Another tank's
  connection and monitoring continued throughout removal.
- Farm workflows, parallel real-video monitoring and Production-mode browser
  checks passed. Cards open the live view directly; the inspector switch and
  census buttons are removed. Accuracy checks remain inside Edit Tank.
- Notice banners are removed from all templates and generated forms. Essential
  service errors remain in operational statuses and form errors.

All camera/inventory checks used disposable farms, recorded videos or an opt-in
virtual live source. Physical-camera accuracy and all-day operation remain
hardware deployment checks. Real Production and Windows startup were not enabled.

## Restored tank controls — 7 October 2026

- **84 regression tests passed**, including atomic source updates, camera/video
  retention, migration from split camera settings, automatic names/codes with
  concurrent creation and archived codes, private source validation, and unchanged
  mortality reconciliation. Raw-preview tests verify overlays are absent and
  detection runs only once.
- Farm workflow and parallel monitoring browser checks passed with the restored
  in-page inspector. Grid/inspector navigation, independent controls, looping,
  photos, delayed frames, source replacement/failure, farm switching and logout
  retain their behavior. Saving feeding settings preserves a concurrent
  population update.
- Source-picker and display browser checks passed: all dropdown branches, retained
  physical camera and video, field hover help, persistent viewer preferences,
  clean previews, inspector continuity and mobile layout. Two accounts receive
  identical frames from one Production worker while drawing different overlays;
  inventory remains unchanged by display controls.
- The shared Production browser check used the opt-in virtual live source in
  disposable farms. It exercises the real inference and background pipeline;
  it does not establish physical-camera accuracy or all-day hardware reliability.

Tank UI interactions use committed version `d187814` as a reference while retaining
the current farm privacy, fixed ensemble settings and automatic census safeguards.

## Production monitoring — 7 October 2026

All automated and browser checks used temporary databases and preview accounts.
Read-only verification confirmed the real installation's Production Mode is off
and no startup installation marker exists. The startup installer was not run.

- **74 regression tests passed** in the full suite. Coverage includes the earlier
  account, inference, farm and parallel monitoring tests, plus fresh-frame census
  stability, visibility gates, conservative zero counts, automatic population and
  feeding changes, estimated versus confirmed mortality, count recovery, and
  matching dispersals and transfers entered before or after camera changes.
- Inventory tests verify duplicate and concurrent observations do not repeat
  deductions. An accelerated 150-observation endurance test preserves population
  and ledger totals across retries and recoveries. Late classifications retain
  the original evidence date; unknown denominators remain N/A.
- Supervisor and WebSocket tests verify monitoring and inventory updates continue
  without viewers, after viewer disconnect and logout, and across server restart.
  Camera failure and per-tank pause do not stop another tank. Saved run/pause
  settings are restored; disabling Production Mode releases workers.
- Native Windows runtime tests verify two independently spawned capture processes
  open and release generated video fixtures, disabled boot exits without creating
  data or launching a server, and the Windows process job terminates the owned
  server and its camera descendant. These do not use physical cameras.
- Production browser checks passed at desktop and 390 px mobile sizes: Admin-only
  mode controls, live-only source restrictions, retained demonstration videos,
  dashboard graphs, camera tiles in Tank Management, removed stocking/check
  buttons, and tooltip help on hover, keyboard focus and touch-and-hold.
- Farm workflow browser checks passed, including the population edit race:
  changing feeding settings does not overwrite a population update received
  while the form is open.
- Parallel monitoring browser checks passed after correcting a photo-counting
  JavaScript error. Both video feeds continued through independent and rapid
  toggles, enlargement, looping, navigation, refresh, photo counting, delayed
  frames, failed/replaced sources, mobile display, farm switching and logout.
  Saved population was unchanged in demonstration mode. The final five-second
  preview sample delivered 0.8 and 1.0 annotated updates per second per tank;
  throughput varies with machine load and is not a physical-camera guarantee.
- Python compilation, JavaScript syntax, PowerShell script parsing and Git
  whitespace checks passed. The optional startup task was syntax checked and
  was not registered or started.

Production cameras now belong to the server. The earlier browser-owned lifecycle
applies only outside Production Mode, where counts do not modify inventory.
Windows startup still needs deployment-account verification with Windows signed
out. Physical whole-tank cameras, independent counting accuracy and an all-day
hardware soak remain deployment checks; fixture tests do not establish them.

## Earlier calibration and demonstration evidence — 6 October 2026

All checks use isolated temporary farms; production farm records were not edited.

- **38 regression tests passed:** weighted daily mortality, historical population,
  zero versus missing checks, legacy migration, stock and feed transactions,
  transfers, private accounts/media/exports, fixed settings, independent counting
  sessions, cancelled-video cleanup, complete CSV history beyond 1,000 records,
  parallel video loops, crossing resets, independent stopping, unassigned/failed
  sources, camera-disconnect isolation, and cancellation waiting for native reads
  before capture cleanup.
- **Desktop and 390 px mobile browser checks passed:** tank creation, stocking,
  mortality checks, feed requirements and planning summaries,
  dispersal sales, filtered reports, Admin account creation, and anonymous
  benchmarks. No JavaScript errors or horizontal page overflow were found.
- **Real parallel monitoring browser checks passed:** two simultaneous videos,
  rapid per-tank toggles, enlarged views sharing the original connection, video looping,
  navigation while monitoring, refreshing without reconnecting, delayed frames,
  photo counting beside another active feed, broken-source isolation and
  replacement, mobile layout, farm switching and
  sign-out. Saved stock was unchanged. Physical multi-camera hardware was not
  exercised by these video fixtures.
- Five-second warm browser measurements delivered approximately **2.4–8.8 annotated
  updates per second per tank** across runs with both recorded videos active on the RTX 2050.
  This measures processed previews, not the original videos' frame rate or a
  guarantee for more cameras. Details are in the local browser QA JSON artifact.
- JavaScript syntax and Git whitespace checks passed.
- A further focused media test passed after rejecting unsupported upload types.

Calibration used 101 labeled validation images, then 10 test images after
selection. Settings minimize validation count MAE over the tested grid, with
box F1 and latency as tie-breakers. The matching IoU for evaluation is 0.50;
this is separate from the fixed suppression settings below.

| Anonymous engine | Confidence | Suppression IoU | Validation count MAE | Test count MAE |
| ---              | ---:       | ---:            | ---:                 | ---:           |
| Engine A | 0.30 | 0.30 | 2.188119 | 1.2 |
| Engine B | 0.35 | 0.30 | 1.990099 | 1.1 |
| Engine C | 0.25 | 0.30 | 2.118812 | 0.6 |
| Combined Counting | Per-engine values above | 0.30 across engines | 2.980198 | 0.5 |

Combined Counting is the requested default. It did not outperform every single
engine on validation. All validation and test source videos also occur in the
training split, so these results do not establish accuracy on independent farm
footage. The profile records annotation and weight fingerprints.

Real inference on the NVIDIA GeForce RTX 2050 counted 20 fish in a labeled image
containing 20, in all six frames of a generated video, and in 30 live camera
updates. Saved population did not change. Across two runs, warm image requests
took approximately 0.23–0.59 seconds; live counting reported 4.4–7.5 updates per
second. The first image request took 6.9–10 seconds while inference warmed up.
The live fixture repeats the same labeled frame and is a pipeline check, not a
motion-tracking accuracy study.

Screenshots are stored locally under `private_data/browser_qa`. Setup and the
repeatable test commands are in `README.md`.
