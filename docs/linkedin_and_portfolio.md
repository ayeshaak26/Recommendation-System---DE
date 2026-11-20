# LinkedIn post and portfolio description

Replace `[LIVE DEMO LINK]` and `[GITHUB LINK]` before publishing. Every number below comes from the
project's own runs (see the README's Results section), so update them if you re-run the pipeline.

---

## LinkedIn post

**Main version (about 1,300 characters)**

Most recommender projects stop at a notebook trained once on a static CSV. I wanted to build the part that comes after: keeping the data and the model fresh without touching them.

So I built a movie recommender with an automated pipeline behind it.

What it does:
🎬 Pick a movie, get 10 similar ones with posters, ratings and trailers
🔄 A scheduled pipeline pulls new and changed movies from the TMDB API
🧹 Bad records are quarantined instead of crashing the run
🧠 The model is rebuilt only when the catalogue actually changes

What I learned building it:

1. The simple model won. I compared TF-IDF against sentence embeddings (all-MiniLM-L6-v2), scored against TMDB's own recommendations for 300 films. TF-IDF came out ahead: precision@10 of 0.26 vs 0.22. Exact matches on director, cast and franchise carry a lot of signal.

2. "Incremental" is not always small. TMDB flagged about 2,200 of my 5,000 movies as changed in a single day. Caching embeddings per movie meant only 461 of 5,437 had to be re-encoded.

3. Idempotency pays for itself. Every stage can be re-run safely, so the fix for almost any failure is "run it again".

4. Monitoring is part of the product. Every run is logged to a table, and a notebook produces a dated data quality report with pass/warn/fail checks.

Stack: Python, pandas, SQLite, pandera, scikit-learn, sentence-transformers, papermill, Streamlit, Windows Task Scheduler. 72 automated tests.

Live demo: [LIVE DEMO LINK]
Code and write-up: [GITHUB LINK]

This product uses the TMDB API but is not endorsed or certified by TMDB.

#DataEngineering #MachineLearning #RecommenderSystems #Python #Streamlit #ETL

---

**Short version (about 500 characters)**

I built a movie recommender that keeps itself up to date.

A scheduled pipeline pulls new and changed movies from the TMDB API, validates them, loads them into SQLite and rebuilds the model only when the catalogue changes. A Streamlit app serves the results with posters and trailers.

One surprise: plain TF-IDF beat sentence embeddings when scored against TMDB's own recommendations (precision@10 of 0.26 vs 0.22).

Demo: [LIVE DEMO LINK]
Code: [GITHUB LINK]

#DataEngineering #MachineLearning #Python

---

**Posting tips**

- Attach a short screen recording (search a movie, open a trailer, switch method) or the recommendations screenshot from `docs/images/`. Posts with a visual get read.
- LinkedIn shows only the first two or three lines before "see more", so keep the opening as it is.
- Put the links in the post body as written, or in the first comment if you prefer; both are common.

---

## Portfolio description

**Title**

Movie Recommender with an Automated Data Pipeline

**One-line summary**

A content-based movie recommendation app fed by a scheduled, self-monitoring ETL pipeline on the TMDB API.

**Card text (about 40 words)**

Search any of 5,400+ movies and get ten similar ones with posters and trailers. Behind the app, a scheduled pipeline pulls new data from TMDB, validates it, and rebuilds the model only when the catalogue changes.

**Full description**

**The problem.** Recommender demos are usually trained once on a static file. They go stale as new films are released, they assume clean data, and nobody can tell whether they still work.

**What I built.** A small data product in three parts:

- **A pipeline** that extracts movies from the TMDB API incrementally (new, changed, or older than 30 days), validates them with pandera, quarantines bad rows, loads them into SQLite with idempotent upserts, and records every run.
- **Two recommenders** built from genres, keywords, cast, director and plot: TF-IDF and sentence embeddings (all-MiniLM-L6-v2), both with cosine similarity. The model is rebuilt only when the set of movies changes, and embeddings are cached so only new texts are encoded.
- **A Streamlit app** with search, a poster grid, ratings, and trailers, which reads only the prepared data and never calls the API.

**Results.**

- 5,437 movies, with 96% trailer coverage and over 99.9% poster coverage.
- TF-IDF beat embeddings against TMDB's own recommendations: precision@10 of 0.26 vs 0.22, and a hit rate of 82% vs 73%.
- A next-day incremental run processed 2,699 movies in about 5.5 minutes and re-encoded only 461 of 5,437 texts.
- 72 automated tests; every pipeline stage is safe to re-run.

**What I would do next.** Blend the two methods, add a collaborative signal from a ratings dataset, and move the schedule from my PC to GitHub Actions so the live app refreshes itself.

**Tech stack**

Python · pandas · SQLite · pandera · requests + tenacity · scikit-learn · sentence-transformers · papermill · Streamlit · pytest · Windows Task Scheduler

**Links**

- Live demo: [LIVE DEMO LINK]
- Source code: [GITHUB LINK]

**Attribution (keep this on the portfolio page)**

This product uses the TMDB API but is not endorsed or certified by TMDB.
