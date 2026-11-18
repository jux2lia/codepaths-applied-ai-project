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
