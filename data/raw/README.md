# Raw data

Download and extract the CC0
[Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
into the project-level `archive/` directory.

Required for catalog generation:

- `anime.csv`
- `anime_with_synopsis.csv`

Required for collaborative training:

- `rating_complete.csv`

The much larger `animelist.csv` and the example HTML folders are not needed by the
current pipeline. Raw files are ignored by Git; publish only the checksummed processed
artifacts to the companion Hugging Face Dataset repository.
