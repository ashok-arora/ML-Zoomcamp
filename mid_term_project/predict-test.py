import requests
import json

flask_url = "127.0.0.1:5000"
docker_url = "127.0.0.1:7000"
heroku_url = "https://midterm-project-spotify.herokuapp.com/predict"

song = {
    "danceability": 0.369,
    "energy": 0.94,
    "key": 9,
    "loudness": -3.6719999999999997,
    "mode": 1,
    "speechiness": 0.086,
    "acousticness": 0.00048499999999999997,
    "instrumentalness": 0.0006389999999999999,
    "liveness": 0.14,
    "valence": 0.517,
    "tempo": 110.021,
    "duration_ms": 148867,
    "time_signature": 4,
    "chorus_hit": 31.05358,
    "sections": 7
}

print(requests.post(heroku_url, json=song).json())
