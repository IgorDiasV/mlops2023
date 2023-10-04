import os
import json
import logging
import requests
import xmltodict

from airflow.decorators import dag, task
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook

import pendulum
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000

logging.basicConfig(level=logging.INFO)

@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary():
    "If it doesn't already exist, create a table to store the episode"

    create_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    @task()
    def get_episodes():
        data = requests.get(PODCAST_URL, timeout=20)
        feed = xmltodict.parse(data.text)
        episodes = feed["rss"]["channel"]["item"]
        logging.info(f"Found {len(episodes)} episodes.")
        return episodes

    podcast_episodes = get_episodes()
    create_database.set_downstream(podcast_episodes)

    @task()
    def load_episodes(episodes):
        logging.info("starting to load the episode")
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []
        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append([episode["link"], episode["title"],
                                     episode["pubDate"],
                                     episode["description"],
                                     filename])

        hook.insert_rows(table='episodes',
                         rows=new_episodes,
                         target_fields=["link",
                                        "title",
                                        "published",
                                        "description",
                                        "filename"])
        logging.info("finished load")
        return new_episodes

    new_episodes = load_episodes(podcast_episodes)

    @task()
    def download_episodes(episodes):
        audio_files = []
        for episode in episodes:
            name_end = episode["link"].split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(EPISODE_FOLDER, filename)
            if not os.path.exists(audio_path):
                logging.info(f"Downloading {filename}")
                audio = requests.get(episode["enclosure"]["@url"], timeout=20)
                with open(audio_path, "wb+") as file:
                    file.write(audio.content)
            audio_files.append({
                "link": episode["link"],
                "filename": filename
            })
        return audio_files

    audio_files = download_episodes(podcast_episodes)

    @task()
    def speech_to_text(audio_files, new_episodes):
        hook = SqliteHook(sqlite_conn_id="podcasts")
        query = "SELECT * from episodes WHERE transcript IS NULL;"
        untranscribed_episodes = hook.get_pandas_df(query)

        model = Model(model_name="vosk-model-en-us-0.22-lgraph")
        rec = KaldiRecognizer(model, FRAME_RATE)
        rec.SetWords(True)

        for index, row in untranscribed_episodes.iterrows():
            logging.info(f"Transcribing {row['filename']}")
            filepath = os.path.join(EPISODE_FOLDER, row["filename"])
            mp3 = AudioSegment.from_mp3(filepath)
            mp3 = mp3.set_channels(1)
            mp3 = mp3.set_frame_rate(FRAME_RATE)

            step = 20000
            transcript = ""
            for i in range(0, len(mp3), step):
                logging.info(f"Progress: {i/len(mp3)}")
                segment = mp3[i:i+step]
                rec.AcceptWaveform(segment.raw_data)
                result = rec.Result()
                text = json.loads(result)["text"]
                transcript += text
            hook.insert_rows(table='episodes',
                             rows=[[row["link"], transcript]],
                             target_fields=["link", "transcript"],
                             replace=True)

podcast_summary()