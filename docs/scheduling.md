# Scheduling the pipeline with Windows Task Scheduler

The pipeline is started by [`run_pipeline.bat`](../run_pipeline.bat). The batch file:

1. moves to the project folder (Task Scheduler starts in `C:\Windows\System32`),
2. activates the `.venv` virtual environment,
3. runs `python run_pipeline.py --stage all`,
4. appends everything printed to `logs\scheduler.log`,
5. runs `python -m pipeline.report`, which uses papermill to execute
   `notebooks\02_data_quality_monitor.ipynb` and save a dated copy such as
   `notebooks\reports\02_data_quality_monitor_2026-11-21.ipynb`,
6. returns the pipeline's exit code (`0` = success, `1` = failed, `2` = virtual environment missing).
   A failed report is logged but does not change the exit code.

The paths below use an example project folder. Replace it with the folder where you cloned the project.

```
C:\Projects\Movie Recommender
```

## 1. Test it by hand first

Always confirm the batch file works before scheduling it. From any folder:

```powershell
& "C:\Projects\Movie Recommender\run_pipeline.bat"; "Exit code: $LASTEXITCODE"
```

Then look at the end of the log:

```powershell
Get-Content "C:\Projects\Movie Recommender\logs\scheduler.log" -Tail 20
```

You should see `Pipeline run succeeded:` followed by the row counts, and
`Pipeline finished with exit code 0`.

## 2a. Create the task in the GUI

1. Press **Win**, type **Task Scheduler**, press **Enter**.
2. In the right-hand panel click **Create Task...** (not "Create Basic Task" - it hides options you need).
3. **General** tab
   - Name: `TMDB Recommender Pipeline`
   - Description: `Daily refresh of the movie database and recommender model.`
   - Select **Run only when user is logged on**. This is the simplest option and needs no password.
     Choose "Run whether user is logged on or not" only if the PC is usually locked or logged out
     at the scheduled time; Windows will then ask for your account password when you save.
   - Leave "Run with highest privileges" **unticked**. The pipeline does not need admin rights.
4. **Triggers** tab -> **New...**
   - Begin the task: **On a schedule**
   - Settings: **Daily**, start time e.g. **09:00**, recur every **1** day
   - Make sure **Enabled** is ticked -> **OK**
5. **Actions** tab -> **New...**
   - Action: **Start a program**
   - Program/script (keep the quotes, the path contains spaces):
     `"C:\Projects\Movie Recommender\run_pipeline.bat"`
   - Add arguments: leave empty (or `--full-refresh` for a task that re-fetches everything)
   - Start in: `C:\Projects\Movie Recommender` (**no** quotes in this box)
   - **OK**
6. **Conditions** tab
   - Untick **Start the task only if the computer is on AC power** if this is a laptop.
   - Tick **Start only if the following network connection is available** -> **Any connection**.
7. **Settings** tab
   - Tick **Run task as soon as possible after a scheduled start is missed**
     (so a run still happens if the PC was off or asleep at 09:00).
   - Tick **Stop the task if it runs longer than** -> **2 hours**.
   - "If the task is already running": **Do not start a new instance**.
8. Click **OK** to save.

## 2b. Or create it with one command

Run this in PowerShell or Command Prompt. It creates the same daily 09:00 task
(`/F` overwrites a task of the same name).

```powershell
schtasks /Create /TN "TMDB Recommender Pipeline" /TR "\"C:\Projects\Movie Recommender\run_pipeline.bat\"" /SC DAILY /ST 09:00 /F
```

A weekly schedule instead (every Monday at 09:00):

```powershell
schtasks /Create /TN "TMDB Recommender Pipeline" /TR "\"C:\Projects\Movie Recommender\run_pipeline.bat\"" /SC WEEKLY /D MON /ST 09:00 /F
```

`schtasks` cannot set the "run after a missed start" and "do not start a new instance" options;
tick them afterwards in the GUI (step 7 above) if you want them.

## 3. Test the scheduled task

Start it immediately, without waiting for the schedule:

```powershell
schtasks /Run /TN "TMDB Recommender Pipeline"
```

Check its status and result:

```powershell
schtasks /Query /TN "TMDB Recommender Pipeline" /V /FO LIST
```

Look at these lines:

| Line | Meaning |
|---|---|
| `Status: Running` | the pipeline is still going |
| `Last Result: 0` | the last run succeeded |
| `Last Result: 1` | the pipeline failed - see `logs\pipeline.log` |
| `Last Result: 2` | the batch file could not find `.venv` |
| `Last Result: 267009` | still running (0x41301) |
| `Last Result: 267011` | the task has never run (0x41303) |

Every run also adds a row to the `pipeline_runs` table, whether it succeeded or failed:

```powershell
& "C:\Projects\Movie Recommender\.venv\Scripts\python.exe" -c "import sqlite3; [print(r) for r in sqlite3.connect(r'C:\Projects\Movie Recommender\data\curated\movies.db').execute('SELECT run_id, started_at, stage, status, rows_extracted, rows_loaded, model_refreshed FROM pipeline_runs ORDER BY run_id DESC LIMIT 5')]"
```

To remove the task:

```powershell
schtasks /Delete /TN "TMDB Recommender Pipeline" /F
```

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Task shows **Last Result 0x1**, `scheduler.log` has new lines | The pipeline itself failed | Read the traceback at the end of `logs\pipeline.log`. The `pipeline_runs` row has the error message too. Fix and re-run - every stage is safe to repeat. |
| **Last Result 0x1**, nothing new in `scheduler.log` | The path to the `.bat` file is wrong or not quoted | In the Action, put the full path in double quotes. The folder name contains spaces. |
| **Last Result 0x2** | `.venv` was not found | Recreate it in the project folder: `python -m venv .venv`, then `.venv\Scripts\pip install -r requirements-pipeline.txt`. |
| `TMDB_API_KEY is not set` in the log | `.env` is missing or still has the placeholder | Put the key in `.env` in the project folder. The pipeline loads it from the project folder, not the current folder, so Task Scheduler needs no extra setup. |
| `TMDB rejected the API key (401)` | Key was mistyped or revoked | Check the key on the TMDB API settings page and update `.env`. |
| Task never starts at the scheduled time | PC was asleep, off, or on battery | Tick "Run task as soon as possible after a scheduled start is missed" and untick the AC-power condition. To wake a sleeping PC, tick "Wake the computer to run this task" on the Conditions tab. |
| Task runs when started by hand but not on schedule | "Run only when user is logged on" and nobody was logged on | Stay logged in (locking the screen is fine), or switch to "Run whether user is logged on or not". |
| `Access is denied` when creating the task | Creating tasks for another user or with highest privileges | Create it for your own account without "highest privileges", or open the terminal as administrator. |
| `DLL load failed ... Application Control policy has blocked this file` | Windows Smart App Control blocked a Python library | Reinstall the exact versions in `requirements-pipeline.txt` (`pyarrow` is pinned to a version that is allowed). Do not turn Smart App Control off for this. |
| Network or timeout errors in the log | No connection when the task ran | The extract stage retries with backoff and then fails cleanly. The next run picks up where it stopped. Add the "network connection available" condition. |
| A run seems stuck in status `running` in `pipeline_runs` | The process was killed (shutdown, task timeout) | Harmless. Start the pipeline again; the same day's run resumes and skips files already downloaded. |
| Two runs overlap | A manual run was started while the scheduled one was going | Set "If the task is already running: Do not start a new instance". |

## The dated data quality report

Each scheduled run leaves a report in `notebooks\reports\`. Open the newest one in Jupyter or VS Code
and scroll to **8. Automated checks** for the pass / warn / fail list. To produce one by hand:

```powershell
& "C:\Projects\Movie Recommender\.venv\Scripts\python.exe" -m pipeline.report --html
```

`--html` also writes a `.html` copy (code hidden) that opens in any browser. Other options:
`--date 2026-11-20` to label the report, `--notebook 01_pipeline_runner -p STAGE load` to execute the
runner notebook with a parameter. The reports folder is git-ignored.

If the log shows `No module named papermill` or `nbconvert`, reinstall the requirements:
`.venv\Scripts\pip install -r requirements-pipeline.txt`.

## What a normal daily run does

Most days only a few hundred movies are new or changed, so a run takes a few minutes:

1. **Extract** - finds new movies, movies TMDB reports as changed, and movies not refreshed for 30 days.
2. **Validate + transform** - rejects bad rows to `data\quarantine\`, writes clean Parquet to `data\staged\`.
3. **Load** - upserts into `data\curated\movies.db`.
4. **Refresh** - rebuilds the recommender only if the set of movies changed.
5. **Log** - one row in `pipeline_runs`, details in `logs\pipeline.log`.
