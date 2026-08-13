# Dataset Attribution

Anime Compass primarily uses:

- Dataset: Anime Recommendation Database 2020
- Author: Hernan Valdivieso
- Source: https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020
- License: CC0 1.0 Universal Public Domain Dedication
- License URL: https://creativecommons.org/publicdomain/zero/1.0/

The source provides 17,562 anime records and anonymous completed-title ratings. The
quality-checked local snapshot contains 57,633,278 unique user/anime ratings from
310,059 users. Adult/Rx titles are excluded by default.

## Migration And Enrichment

`archive/anime.csv` and `archive/anime_with_synopsis.csv` are the primary metadata
sources. `archive/rating_complete.csv` trains the compact collaborative item
embeddings used at runtime.

The archive stops at 2022 and does not contain posters, character/cast relationships,
or complete staff credits. During the one-time migration, the former CC0
`Anime Dataset Top 10K Normalized` snapshot by Uday Kumar was used only to preserve
those fields for matching MAL IDs and to retain newer catalog entries. Its raw files
are no longer required by the application.

Generated artifacts are hosted separately in a Hugging Face Dataset repository and
verified against `data/artifacts.manifest.json` before use.
