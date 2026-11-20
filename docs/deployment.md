# Deploying to a subdomain of ayeshakhan.tech

## Why the app cannot run on the current hosting

`ayeshakhan.tech` is on shared web hosting (Apache with a DirectAdmin control panel). That kind of
hosting serves files and PHP. A Streamlit app is a Python server that stays running and talks to the
browser over a websocket, which shared hosting does not allow.

So the app runs on a host built for it, and the subdomain points visitors there. Two ways to do that
are described below.

| | Option A: Streamlit Cloud + embed page (recommended) | Option B: Render with a real custom domain |
|---|---|---|
| Cost | Free | Free tier |
| Address visitors see | `movies.ayeshakhan.tech` | `movies.ayeshakhan.tech` |
| How | A one-file page on your hosting shows the app in a full-window frame | A DNS record points the subdomain straight at the app |
| Changes on your hosting | Create subdomain, upload one file | Add one CNAME record |
| When idle | Sleeps after several days without visitors; one click wakes it | Sleeps after 15 minutes; the next visitor waits about a minute |
| Extra files needed | None (already in this repo) | A start command (below) |

Both start the same way: the code has to be on GitHub.

## Step 1: put the project on GitHub

1. Create a new **empty** repository on GitHub (no README, no .gitignore). Public is required for the
   free Streamlit tier.
2. In the project folder:

```powershell
git init -b main
```

```powershell
git add .
```

```powershell
git status
```

Check the list before committing: `.env` must **not** appear, and `data/curated/` **must** appear
(the app reads it). Then:

```powershell
git commit -m "Movie recommender with automated TMDB pipeline"
```

```powershell
git remote add origin https://github.com/<your-username>/<your-repo>.git
```

```powershell
git push -u origin main
```

## Option A: Streamlit Community Cloud + embed page

### A1. Deploy the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → pick the repository, branch `main`, main file `app.py`.
3. Optionally choose a custom address such as `ayesha-movie-recommender`.
4. **Deploy**. The first build takes a few minutes.

No secrets are needed: the app reads only `data/curated/` and never calls the TMDB API.

Your app is now at `https://<app-name>.streamlit.app`.

### A2. Create the subdomain

In the DirectAdmin panel for ayeshakhan.tech:

1. **Account Manager → Subdomain Management → Add Subdomain**, name it `movies`.
   This creates the folder `domains/ayeshakhan.tech/public_html/movies/`.
2. Make sure HTTPS covers it: **Account Manager → SSL Certificates → Let's Encrypt**, tick
   `movies.ayeshakhan.tech` and save. The page must be served over HTTPS or browsers will block the frame.

### A3. Upload the embed page

1. Open [`deploy/subdomain/index.html`](../deploy/subdomain/index.html) and replace `YOUR-APP-NAME`
   in **both** places with your app's name.
2. Upload it as `index.html` into the `movies` folder (File Manager or FTP).
3. Visit `https://movies.ayeshakhan.tech`.

**If you prefer a plain redirect** instead of the frame (the address bar then changes to
`streamlit.app`), put this in a file named `.htaccess` in the `movies` folder instead of `index.html`:

```apache
Redirect 302 / https://YOUR-APP-NAME.streamlit.app/
```

### A4. Things to know about the embed

- The app sleeps after several days with no visitors. A visitor then sees a "wake up" button inside
  the frame; it takes about half a minute.
- Trailers play inside the frame. If full-screen video is blocked in some browser, use the
  "Watch on YouTube" link in the player.
- Shared links always go to the home page, because the frame does not change the address bar.

## Option B: Render with a real custom domain

Use this if you want the subdomain to be the app's actual address with no frame.

1. Sign in at <https://render.com> with GitHub → **New → Web Service** → pick the repository.
2. Settings:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command:
     `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Instance type: **Free**
3. Add an environment variable `PYTHON_VERSION` = `3.13.9` so the build uses the same Python as
   development.
4. After the first deploy, open **Settings → Custom Domains → Add** `movies.ayeshakhan.tech`.
   Render shows a target such as `<service>.onrender.com`.
5. In DirectAdmin: **Account Manager → DNS Management → Add Record**, type **CNAME**, name `movies`,
   value the Render target (with a trailing dot if the panel asks for one). Do **not** also create
   the subdomain in Subdomain Management, or its A record will conflict.
6. Wait for DNS (minutes to an hour). Render issues the HTTPS certificate automatically.

## Keeping the live app up to date

The pipeline runs on your PC. The hosted app shows whatever is in `data/curated/` on GitHub, so after
a pipeline run:

```powershell
git add data/curated
```

```powershell
git commit -m "Refresh movie data"
```

```powershell
git push
```

Both Streamlit Cloud and Render redeploy automatically on push. Pushing once a week is plenty; each
push adds roughly 30 MB to the repository's history, so pushing daily would make the repository large
over time.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Blank frame on the subdomain | `YOUR-APP-NAME` not replaced, or the page is served over `http://` | Edit `index.html`; enable the Let's Encrypt certificate for the subdomain |
| "The movie database has not been built yet" on the live app | `data/curated/` was not committed | `git add data/curated`, commit, push |
| Build fails on `pyarrow==22.0.0` | The host's Python version has no wheel for that version | Set the Python version to 3.13 in the host's settings |
| App works on `streamlit.app` but not in the frame | A browser extension or strict privacy mode blocks third-party frames | Use the redirect variant in A3 |
| Subdomain shows the hosting's default page | `index.html` is in the wrong folder | It must be directly inside `public_html/movies/` |
