from model import Net
from utils import CustomDataLoader, EarlyStopper
import torch
import tqdm
import numpy as np
import pandas as pd

def train_one_epoch(model, train_dataloader):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(train_dataloader.loader, desc="train", smoothing=0, mininterval=1.0)
    for i, data in enumerate(tk0):
        data = {k: v.to(device) for k, v in data.items()}  #tensor to GPU
        loss = model.cal_loss(data["team1"], data["team2"], data["team1_adv"], data["team1_res"],
                            data["team2_adv"], data["team2_res"], data["tour"], data["country"],
                            data["neutral"], data["t1_status"])
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            tk0.set_postfix(loss=total_loss / 10)
            total_loss = 0

def valid_one_epoch(model, valid_dataloader):
    model.eval()
    with torch.no_grad():
        total_loss, count = 0, 0
        tk0 = tqdm.tqdm(valid_dataloader.loader, desc="validation", smoothing=0, mininterval=1.0)
        for i, data in enumerate(tk0):
            data = {k: v.to(device) for k, v in data.items()}
            batch_size = int(list(data.values())[0].shape[0])
            count += batch_size
            loss = model.cal_loss(data["team1"], data["team2"], data["team1_adv"], data["team1_res"],
                            data["team2_adv"], data["team2_res"], data["tour"], data["country"],
                            data["neutral"], data["t1_status"])
            total_loss += (loss.item()*batch_size)
    print(count)
    return total_loss/count

def predict(model, test_dataloader):
    model.eval()
    result = None
    with torch.no_grad():
        tk0 = tqdm.tqdm(test_dataloader.loader, desc="test", smoothing=0, mininterval=1.0)
        for i, data in enumerate(tk0):
            data = {k: v.to(device) for k, v in data.items()}
            pred = model.forward(data["team1"], data["team2"], data["team1_adv"], data["team1_res"],
                            data["team2_adv"], data["team2_res"], data["tour"], data["country"],
                            data["neutral"])
            pred = torch.softmax(pred, dim=-1)
            if result is None:
                result = pred.cpu().detach().numpy()
            else:
                result = np.concatenate([result, pred.cpu().detach().numpy()], axis=0)
    return result

if __name__ == "__main__":
    model = Net()
    train_dataloader = CustomDataLoader(tfrecord_file_path="data/train.tfrecord",
                                index_file_path="data/train.index", mini_batch_size=256)
    valid_dataloader = CustomDataLoader(tfrecord_file_path="data/valid.tfrecord",
                                index_file_path="data/valid.index", mini_batch_size=256)
    test_dataloader = CustomDataLoader(tfrecord_file_path="data/test.tfrecord",
                                index_file_path="data/test.index", mini_batch_size=256)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    es = EarlyStopper(patience=5, mode="min")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch = 500
    for epoch_i in range(epoch):
        print('epoch:', epoch_i)
        train_one_epoch(model, train_dataloader)
        result = valid_one_epoch(model, valid_dataloader)
        print("valid result:", result)
        if es.stop_training(result, model):
            print(f'validation: best loss: {es.best_value}')
            model.load_state_dict(es.best_weights)
            break
    torch.save(model.state_dict(), "data/model.pth") #save best auc model
    model.load_state_dict(torch.load("data/model.pth"))
    print(valid_one_epoch(model, valid_dataloader))
    preds = predict(model, test_dataloader)
    data = pd.read_csv("data/results.csv")
    test_data = data[data["home_score"].isna()].reset_index(drop=True)
    for i in range(len(test_data)):
        test_data.loc[i, "home_team_loss"] = preds[i, 0]
        test_data.loc[i, "home_team_win"] = preds[i, 1]
        test_data.loc[i, "home_team_draw"] = preds[i, 2]
    print(test_data)
    test_data[["home_team", "away_team", "home_team_loss", "home_team_win", "home_team_draw"]].to_csv("data/pred.csv")



        
        



    