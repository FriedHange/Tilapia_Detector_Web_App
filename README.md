# AquaTrack farm management

The app has protected Admin and farmer User workspaces. Each farmer receives a
private farm; only Admin can see or manage multiple farms. Existing records are
retained in the Admin-owned **Existing farm**.

## Start and create the first administrator

Run `run_app.bat` as before, or:

```powershell
venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Before signing in for the first time, create an administrator locally:

```powershell
venv\Scripts\python.exe manage.py create-admin --username admin
```

The command prompts for a password of at least 12 characters. There are no
bundled production credentials. Alternatively, set `TILAPIA_ADMIN_USERNAME` and
`TILAPIA_ADMIN_PASSWORD` for the first startup. These variables only bootstrap
an administrator if none exists.

Open `http://localhost:8000/login`. Admin creates farmer accounts in **Farms &
Accounts**, gives each farmer their initial password, and can disable access or
reset passwords there. Farmers must change their initial password at sign-in.
Login sessions expire after eight hours; password changes and account changes
revoke existing sessions.

## Workspaces

- **Dashboard:** population and mortality graphs, feeding estimates, monitoring
  health, coverage and persistent alerts. Estimated losses and confirmed deaths
  appear separately.
- **Tank Management:** parallel camera grid. Click a tank to open its live view
  while other cameras keep running. Each tank has one Start/Stop toggle; camera
  accuracy checks are available in Edit Tank. Stock Fish and Mortality
  Check buttons are removed. Initial population and exceptional corrections are
  available in tank setup; corrections require a reason.
- **Food Management:** daily and seven-day requirements using saved population,
  configured average weight and feeding percentage. Cameras do not measure weight.
- **Dispersal Management:** sales and outgoing transfers to a named recipient.
  Record these before or after moving fish. Recent unexplained camera losses are
  matched chronologically without event selectors. Dispersal references are
  filled automatically; confirmed deaths are reviewed separately.
- **Analytics & Reports:** population, estimated/confirmed mortality, coverage,
  feeding estimates and dispersal records, including CSV exports.
- **Benchmarks:** Admin-only fixed counting evaluations outside Production Mode.

Buttons explain their action on hover or keyboard focus. On touch screens, hold
for half a second to show the explanation without performing the action.
Edit Tank includes help buttons for its fields. New tanks receive a name and
permanent code automatically; names can be changed. Codes include archived tanks
and are assigned atomically when saving.

The **Source** dropdown shows the matching USB camera number, RTSP address,
video upload or saved-video selector. Each tank remembers its physical camera
and its last selected video. Selecting a demonstration video keeps the physical
camera and its census validation. Production hides video choices and uses the
physical camera; switching it off restores the demonstration selection.
Changing the physical camera requires a new validation. Source and tank settings
save together; camera numbers refer to the system PC, not the browser device.

**Display options** control boxes, fish labels/IDs, confidence percentages,
motion trails and center dots. **Clean video** hides every overlay. Preferences
are saved for your account on the current device and apply to cameras, videos
and uploaded photos;
other viewers, counting and inventory are unaffected. The browser draws overlays
from a raw preview plus detection metadata; display changes do not run
additional detection. Reload the app after updating the server to load the new
preview controls. Boxes, labels
and trails default on; percentages and dots default off.

**Delete tank** is available inside Edit Tank. The confirmation shows the latest
saved population. Remaining stock is recorded as tank removal, separately from
deaths and dispersals, then excluded from active inventory and feeding. History
and permanent codes are kept. A population change during confirmation requires
reviewing the updated count. Removed tanks cannot accept new operations.

## Production Mode (initially off)

Admin controls one installation-wide switch from Dashboard or Tank Management.
Enabling it makes all eligible, enabled tanks use live cameras only. Photo counts,
video upload/playback controls and image evaluation are unavailable. Existing
media is retained for demonstration use after the mode is switched off.
Installation notices and startup controls are shown only to Admin. Farmers keep
camera health and validation guidance without a Production Mode panel.

Before enabling automatic updates for a tank:

1. Set its independently verified initial/saved population and feeding inputs.
2. Open **Edit Tank**, then choose **USB camera** or **Network camera** in Source.
   Use a separate camera view per tank; sources cannot be owned by two tanks.
3. In **Edit Tank**, select **Check camera accuracy** and confirm the view covers
   the whole population. Save any pending edits first.
   Keep the view fixed and avoid moving fish during the check.

Validation needs at least 12 fresh observations spanning a full minute. It
measures the counting error band and lighting, contrast, sharpness and coarse
scene reference. At least 90% of counts must be within 5% of known population
(minimum one fish). Poor or inconsistent views fail validation.

Production uses non-overlapping 60-second census windows, at least 12 fresh
samples per window, and three consistent windows before reconciling changes.
Changes inside the measured error band are held; small losses may therefore
remain unresolved. A reliable zero population requires ten consistent windows.
Bad visibility, stale/frozen frames and outages reset the evidence window rather
than recording deaths. Physical-camera changes, population corrections and calibration
changes require fresh validation.

Unexplained sustained decreases update saved population and **estimated
mortality**. Increases first correct eligible earlier undercounts, then record
additional fish. These are inventory estimates, not classifications of a fish's
cause of death. Feed and biomass estimates update with saved population.

Dashboard and Analytics offer an optional **Review estimated losses** action when
losses exist. Confirming deaths reclassifies the camera evidence without another
population deduction; it is not a daily check required to keep monitoring active.

Dispersal or confirmed mortality entered after a camera loss reclassifies the
matched quantity without subtracting it twice. A movement entered first remains
expected until the camera observes the population change. Automatic dispersal
matching covers 24 hours and consumes recent unexplained losses chronologically.
The simplified form records transfers as outgoing fish only; it does not credit
another tank. Existing internal-transfer records and API support are retained.
The separate confirmed-death review can select older evidence. Overdue expected movements hold reconciliation until population is
verified and the camera revalidated. Every change and correction retains evidence.

Production cameras belong to the server, with one worker per farm/tank. Viewers
share its pipeline; closing the page, changing farms, signing out, screen locking
or login expiry does not stop it. Explicit Start/Stop preferences persist across
server restarts. Turning Production Mode off stops production workers and retains
preferences. No camera starts automatically without current census validation.

A stopped camera reconnects with backoff from one second to one minute. Capture
runs in disposable child processes so a blocked driver can be terminated. Shared
inference uses a fair queue and newest-frame capture. Actual throughput depends
on hardware; slow/stale observations cannot update inventory.

## Unattended Windows startup (optional; not activated)

`run_app.bat` performs setup and then uses the supervised runner (without reload).
It skips opening a browser when Production Mode is already on. To run without
the setup/browser step,
use the already-installed environment:

```powershell
powershell -NoProfile -File scripts/run-production.ps1
```

This runner uses one Uvicorn worker, no reload, no dependency installation and no
browser launch. It restarts a crashed server or one whose heartbeat stalls for
three minutes. Operational and console logs rotate under `private_data`.
On Windows, an owned process job also releases the server and its camera child
processes if the runner exits, so a restart can reopen those cameras.

To configure boot support later, run locally:

```powershell
powershell -NoProfile -File scripts/install-startup.ps1
```

The script asks for a Windows account with access to the project and cameras.
Windows stores its credentials; no password is written to application files. It
registers a limited-privilege startup task and does not start it immediately.
The boot runner exits unless Production Mode is on. Use `-Uninstall` to remove the
task. The app never invokes the installer, and implementation/testing does not
register a task or enable Production Mode on the real installation.

Verify camera access under the scheduled account, with Windows signed out, before
relying on boot monitoring. The PC must remain powered and awake; the app does
not change Windows power settings. Physical camera access and all-day operation
must be validated on the deployment hardware.

Outside Production Mode, videos loop as labeled demonstrations, camera feeds
are browser-owned, and counting never changes inventory. Enlarging a view or
switching modules retains those feeds; closing the page or signing out releases
them. Playback loops reset crossing counters.

## Mortality and units

Daily loss rate = estimated losses plus confirmed deaths divided by opening
population plus incoming fish that day, multiplied by 100. The two classifications
remain separate in reports. Dispersals are excluded from mortality. Recovered
counts reverse prior estimates; later classifications use the original evidence
date. Farm/period rates use recorded fish-days, not averages of tank percentages.

Unobserved days are not zero-death days. Coverage reports reliable and total
observation seconds; a monitored current day is labeled partial. Zero or unknown
denominators show N/A. Explicit inventory corrections make that day's denominator
unknown. Legacy openings and losses without sufficient evidence are not invented.
Reporting dates use Asia/Manila; feed uses kilograms, configured weight grams,
and sale prices PHP.

Detailed tank camera/video telemetry expires after 30 days. Daily coverage,
inventory movements, census evidence, dispersals and audits are retained. Alerts
are deduplicated while active and record recovery when resolved.

## Storage and migration

`private_data/accounts.db` stores accounts, farm identities, sessions and audits.
New farms have private database files under `private_data/farms`; media is under
`private_data/media/<farm-id>`. Database paths are generated by the server.
The existing `tilapia_web_analytics.db` remains the legacy farm's database.

Before an existing farm database is first migrated, a SQLite backup is created
alongside it with the `.pre-farm.bak` suffix. Back up the account registry, farm
databases, legacy database and media together. Stop the server before restoring
them. Do not expose `private_data` through a separate web server.

New farms start empty, including after restart. HTTP, exports, media and WebSocket
connections enforce farm access. Camera/file sources are checked against the
selected farm. Demonstration trackers and cached images are separated by session; production trackers are separated by farm/tank;
shared detector inference is serialized to prevent concurrent predictor mutation.

## Calibration

`detection_profile.json` contains the measured, locked settings and evidence.
Recalibrate after changing any bundled detector weights:

```powershell
venv\Scripts\python.exe calibrate.py --dataset "D:\Downloads\DL Manager 2\Compressed\Tilapia Fingerlings.v1i.coco"
```

Calibration reads COCO labels from `valid` and selects by counting MAE, then F1
at matching IoU 0.50, then latency. It evaluates `test` only after selecting
settings. Predictions at minimum confidence are cached locally, and the chosen
settings are confirmed with native inference. Startup checks weight fingerprints;
a changed weight is disabled until its profile is recalibrated. Restart after
calibration to reload the counting service.

All validation and test source videos also appear in training. Results are
dataset-specific and are not evidence of accuracy on independent farm videos.

## Verification

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe -m unittest discover -s tests -v
node --check static/farm.js
node --check static/monitoring.js
node --check static/tooltips.js
```

For browser checks, start `venv\Scripts\python.exe tests/serve_preview.py` in a
separate terminal. It uses disposable databases and **preview-only credentials**;
it never opens production farm records. Then run:

```powershell
venv\Scripts\python.exe tests/browser_smoke.py --executable "PATH_TO_CHROMIUM_EXE"
venv\Scripts\python.exe tests/browser_monitoring.py --executable "PATH_TO_CHROMIUM_EXE"
venv\Scripts\python.exe tests/browser_production.py --executable "PATH_TO_CHROMIUM_EXE"
venv\Scripts\python.exe tests/browser_usability.py --executable "PATH_TO_CHROMIUM_EXE"
venv\Scripts\python.exe tests/real_counting_smoke.py --dataset "PATH_TO_COCO_DATASET"
```

Browser screenshots are saved under `private_data/browser_qa`. The preview
server listens on port 8001; production uses port 8000.

For the two-viewer Production overlay check, restart the preview server with
`tests/serve_preview.py --display-fixture`, then run:

```powershell
venv\Scripts\python.exe tests/browser_tank_controls.py --executable "PATH_TO_CHROMIUM_EXE"
```

This opt-in fixture supplies one virtual live source to the real background
processing pipeline; it opens no physical camera and uses only temporary farms.
