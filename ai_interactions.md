# AI Collaboration Log

A running log of how AI (Claude) helped — or got it wrong — while building this project.
Add an entry **as you go**, not at the end.

Format: date · what I asked · what the AI suggested · what I did (accepted / revised / rejected) · why

---

## 2026-11-18 — Mood from audio features
- **Asked:** The Kaggle dataset has no `mood` column, but my original scorer gave +1 for mood. What do I do?
- **AI suggested:** Derive mood from valence + energy (four quadrants: happy, calm, angry, sad).
- **Decision:** ✅ Accepted. It keeps the original scoring idea and uses data the dataset actually has.
- **Caveat:** These are rough labels — a low-valence/high-energy song isn't always "angry."

## 2026-11-18 — Path planning approach
- **Asked:** How should the agent plan a route from my taste to a new genre?
- **AI suggested:** A full graph search over a song-similarity graph.
- **Decision:** ✏️ Revised. Start with straight-line waypoints between my taste centroid and the destination genre centroid; the Check/Revise loop inserts bridge songs when a step is too big. Graph search can come later if needed.
- **Why:** Simpler to build, debug, and explain, and it gives the Revise step real work to do.

## 2026-11-18 — Taste profile source
- **Asked:** Can I use my own Spotify playlist instead of a hand-written profile?
- **AI suggested:** Export with Exportify and match songs to the Kaggle dataset by Spotify track ID (from the `Track URI` column), logging unmatched songs.
- **Decision:** ✅ Accepted. Newer Exportify exports may not include audio features, so the features come from Kaggle instead.
- **Later correction:** My own export turned out to *include* audio features (see "Using the playlist's own audio features" below), so this assumption was wrong for my file.

## 2026-11-18 — Project setup
- **Asked:** Help set up step 0 (venv, pinned requirements, .gitignore).
- **AI suggested:** Python 3.11 venv (3.14 is installed too, but some scientific packages lag behind new Python releases), pin exact versions, gitignore secrets and my personal playlist export.
- **Decision:** ✅ Accepted.

## 2026-11-18 — Arrival check failing on almost every route
- **Asked:** Why does the agent's first attempt fail `no_arrival` on nearly every route, and what's the point of the route if the destination is fixed from the start?
- **What was wrong:** The AI's first design let the Act step pick the last song by how it *sounds* (compass score) without checking its genre label. In the 50 songs near the destination's centre, most aren't labelled with that genre, so round 0 failed the arrival check on 26/30 routes for my playlist (12/12 for the example playlists), and Revise always applied the same fix. That made the Check → Revise loop look staged.
- **AI suggested:** Make "the last song must be a destination song" part of the Plan, since it's known up front, and keep the arrival check only as a safety net.
- **Decision:** ✅ Accepted, after asking what the purpose of the route becomes. Answer: the Plan guarantees *where* you end up; Check → Revise now only deals with problems that can't be known in advance — jarring jumps (bridge songs inserted), going backwards, repeated artists.
- **Result:** Routes needing revision (default 4 steps) dropped from 26/30 to 9/30 for my playlist, and every remaining revision fixes a real problem. Example: `classic_rock → big room` fails `step_too_big` and gets a bridge song inserted. New edge case: `neo_soul → hip hop` can't be smoothed and hits the revision cap with a warning.

## 2026-11-18 — Which dataset? (low match rate)
- **Asked:** My playlist only matched 32% (6/19 songs). Is that expected?
- **AI found:** Not a bug. 7 of the 13 missing songs are by artists that aren't in the 30k "30000 Spotify Songs" dataset at all (TV Girl, Men I Trust, Yot Club…), and some are newer than the dataset (early 2020). The 6 matches were mostly labelled r&b, which doesn't describe my taste well.
- **AI suggested:** Switch to the larger Kaggle "Spotify Tracks Dataset" (114k tracks, 114 genres incl. indie and lofi).
- **Decision:** ❌ Rejected. I kept the 30k dataset — it's enough for the project, and I didn't want to rework the data layer. Consequence: destinations are its 6 genres / 24 subgenres (no jazz, metal or lofi), so the original "lofi → metal" demo idea became routes like `classic_rock → big room`.

## 2026-11-18 — Act step jumping to far-away songs (debugging)
- **What happened:** Testing on my playlist, routes to destinations close to my taste (neo soul, rap) hit the revision cap with huge jumps.
- **AI found:** The scorer's genre term (weight 2.0, inherited from my Module 3 scorer) outweighed feature closeness, so the agent jumped to any song anywhere with the right "compass" reading; bridges couldn't fix that. Also, "Redbone" came back as a different release (new track ID) of a song already in my playlist.
- **AI suggested:** Only let the scorer choose among the 50 songs nearest each waypoint, aim at the "heart" of the genre (songs that clearly sound like it), and exclude playlist songs by title + artist as well as ID. Then pick `MAX_STEP` by running all 30 destinations instead of guessing (0.10 left 7–9 routes failing; 0.12 passes all 30).
- **Decision:** ✅ Accepted. The Module 3 formula still ranks the songs; it just ranks nearby ones.

## 2026-11-18 — Using the playlist's own audio features (Disney playlist)
- **Asked:** Why does my Disney playlist fail with "0% match"? I thought the CSV was only used for the taste profile.
- **AI found:** The profile needs audio features, and the code gets them by finding each song in the catalog — which has almost no Disney music. But my Exportify export actually *does* contain audio-feature columns (they match the catalog's values within 0.007).
- **AI suggested:** Build the taste profile straight from the CSV's feature columns (so any playlist works), keeping catalog matching as a fallback.
- **Decision:** ❌ Rejected for now — it touches the data loading, profile, agent and guardrails, and I'd rather finish the required parts. Listed under future improvements in the README.

## 2026-11-18 — Bubble chart hid the route (debugging)
- **What happened:** The chart tests passed, but when the AI rendered the chart to an image to look at it, the route was buried under the 28k background dots and squeezed into a tiny patch.
- **Fix (accepted):** Draw every layer with WebGL so the stacking order holds, and start zoomed in on the route (double-click shows the whole map). Lesson: passing tests ≠ a readable chart.

## 2026-11-18 — RAG / lineage stretch goal
- **Plan said:** Use a lineage file of samples, with the demo "Steely Dan → Kanye West's *Champion*".
- **AI found:** Neither song is in the 30k dataset. Of 39+ famous sample pairs checked, 19 have both songs in the catalog. It also found that sample-linked songs usually *sound* far apart (e.g. *Under Pressure* → *Ice Ice Baby* is 0.22, above the 0.12 step limit) — the link is exactly what audio features miss.
- **AI suggested:** Verify every pair against Wikipedia before using it (each edge in `data/lineage.json` stores the article URL and a quote), retrieve candidates from both sources (nearest songs + lineage links to the previous song or my playlist), and allow a bigger step (≤ 0.30) only along a sample link.
- **Decision:** ✏️ Revised the plan's demo to `examples/sampled_classics.csv → rap`: with lineage on, the route goes through 2Pac's *Changes* (samples *The Way It Is*) and Kendrick Lamar's *King Kunta* (interpolates *Smooth Criminal*); with it off, neither appears. Two relationships were relabelled to match the sources exactly ("Stay With Me" is a credited *melodic similarity*, "Mi Gente" a *remake*).
- **Bug caught while collecting README evidence:** steps without a lineage link printed `lineage: nan` (pandas 3 turned `None` into a truthy NaN, which also produced invalid JSON in one trace). Fixed and covered by a test.
