import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import deque, defaultdict
import numpy as np
import random
from tqdm import tqdm
import pickle
from tfrecord import TFRecordWriter
from tfrecord.tools import tfrecord2idx

def encoder_data(data):
    lbe_encoder = {}
    sparse_col = ["home_team", "away_team", "tournament", "city", "country"]
    team_list = list(data["home_team"])+list(data["away_team"])
    lbe_encoder["team"] = LabelEncoder()
    lbe_encoder["team"].fit(team_list)
    data["home_team"] = lbe_encoder["team"].transform(data["home_team"])+1
    data["away_team"] = lbe_encoder["team"].transform(data["away_team"])+1
    print("team", max(data["home_team"].max(), data["away_team"].max()))
    for col in sparse_col[2:]:
        lbe = LabelEncoder()
        lbe_encoder[col] = lbe
        data[col] = lbe.fit_transform(data[col])+1
        print(col, data[col].max())
    return data, lbe_encoder

def gen_data(data):
    train_data = []
    test_data = []
    team_history = defaultdict(deque)
    for index, row in tqdm(data.iterrows()):
        date, home_team, away_team, home_score, away_score, tournament, _, country, neutral = row
        is_test = np.isnan(home_score)
        if is_test:
            test_data.append(
                (home_team, 
                away_team, 
                list(team_history[home_team]), 
                list(team_history[away_team]),
                tournament,
                country,
                neutral,
                -1
                )
            )
            continue
        year = int(date[:4])
        # win:1 lose:0 draw:2
        home_status, away_status = 2, 2
        if home_score>away_score:
            home_status, away_status = 1, 0
        elif home_score<away_score:
            home_status, away_status = 0, 1
        else:
            home_status, away_status = 2, 2

        if year>1900:
            train_data.append(
                (home_team, 
                away_team, 
                list(team_history[home_team]), 
                list(team_history[away_team]),
                tournament,
                country,
                neutral,
                home_status)
            )
            train_data.append(
                (away_team, 
                home_team, 
                list(team_history[away_team]), 
                list(team_history[home_team]),
                tournament,
                country,
                neutral,
                away_status)
            )
        team_history[home_team].append((away_team, home_status))
        team_history[away_team].append((home_team, away_status))
        if len(team_history[home_team])>10:
            team_history[home_team].popleft()
        if len(team_history[away_team])>10:
            team_history[away_team].popleft()
    return train_data, test_data

def convert_tfrecord(data, mode="train"):
    writer = TFRecordWriter(f"data/{mode}.tfrecord")
    for t1, t2, t1_seq, t2_seq, tour, country, neutral, t1_status in data:
        t1_adv = [adv for adv, res in t1_seq]
        t1_res = [res for adv, res in t1_seq]
        t2_adv = [adv for adv, res in t2_seq]
        t2_res = [res for adv, res in t2_seq]
        writer.write({
            "team1": (t1, "int"),
            "team2": (t2, "int"),
            "team1_adv": ([0]*(10-len(t1_adv))+t1_adv[-10:], "int"),
            "team1_res": ([0]*(10-len(t1_res))+t1_res[-10:], "int"),
            "team2_adv": ([0]*(10-len(t2_adv))+t2_adv[-10:], "int"),
            "team2_res": ([0]*(10-len(t2_res))+t2_res[-10:], "int"),
            "tour": (tour, "int"),
            "country": (country, "int"),
            "neutral": (neutral, "int"),
            "t1_status": (t1_status, "int")
        })
    writer.close()
    tfrecord2idx.create_index(tfrecord_file=f"data/{mode}.tfrecord", index_file=f"data/{mode}.index") 





if __name__=="__main__":
    data = pd.read_csv("data/results.csv")
    train_data = data[~data["home_score"].isna()]
    test_data = data[data["home_score"].isna()]
    print(test_data)
    train_data = train_data.sort_values('date')
    data = pd.concat([train_data, test_data])
    data['neutral'] = data['neutral'].map({True:1, False:0})
    data, lbe_encoder = encoder_data(data)
    with open("data/encoder.pkl", "wb") as f:
        pickle.dump(lbe_encoder, f)
    train_data, test_data = gen_data(data)
    random.shuffle(train_data)
    valid_data, train_data = train_data[:7000], train_data[7000:]
    print(len(train_data))
    convert_tfrecord(train_data, mode="train")
    convert_tfrecord(test_data, mode="test")
    convert_tfrecord(valid_data, mode="valid")