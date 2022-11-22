from torch import nn
import torch

class Net(nn.Module):
    def __init__(self, team_nums=320, tournament_nums=140, city_nums=2010, country_nums=270):
        super().__init__()
        self.team_embedding = nn.Embedding(team_nums, 32, padding_idx=0)
        torch.nn.init.normal_(self.team_embedding.weight, 0, 0.0001)
        self.tournament_embedding = nn.Embedding(tournament_nums, 32, padding_idx=0)
        torch.nn.init.normal_(self.tournament_embedding.weight, 0, 0.0001)
        #self.city_embedding = nn.Embedding(city_nums, 32, padding_idx=0)
        self.country_embedding = nn.Embedding(country_nums, 32, padding_idx=0)
        torch.nn.init.normal_(self.country_embedding.weight, 0, 0.0001)
        self.neutral_embedding = nn.Embedding(5, 32)
        torch.nn.init.normal_(self.neutral_embedding.weight, 0, 0.0001)
        self.status_embedding = nn.Embedding(5, 32)
        torch.nn.init.normal_(self.status_embedding.weight, 0, 0.0001)
        self.gru = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(7*32, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            #nn.Softmax(dim=-1)
        )
        self.loss = nn.CrossEntropyLoss()
        self.dcn = CrossNetwork(7*32, 2)
        self.dcn_mlp = nn.Linear(7*32, 3)
    
    def forward(self, t1, t2, t1_adv, t1_res, t2_adv, t2_res, tour, country, neut):
        t1e = self.team_embedding(t1.squeeze(1))
        t2e = self.team_embedding(t2.squeeze(1))
        t1_adve = self.team_embedding(t1_adv)
        t1_rese = self.status_embedding(t1_res)
        t2_adve = self.team_embedding(t2_adv)
        t2_rese = self.status_embedding(t2_res)
        toure = self.tournament_embedding(tour.squeeze(1))
        countrye = self.country_embedding(country.squeeze(1))
        neute = self.neutral_embedding(neut.squeeze(1))
        # gru
        t1_seq = torch.concat([t1_adve, t1_rese], dim=-1)
        t2_seq = torch.concat([t2_adve, t2_rese], dim=-1)
        seq_output1, hn1 = self.gru(t1_seq)
        seq_output2, hn2 = self.gru(t2_seq)
        #mlp
        inputs = torch.concat([t1e, t2e, hn1.squeeze(0), hn2.squeeze(0), countrye, toure, neute], dim=-1)
        output_1 = self.mlp(inputs)
        output_2 = self.dcn_mlp(self.dcn(inputs))
        return output_1+output_2

    def cal_loss(self, t1, t2, t1_adv, t1_res, t2_adv, t2_res, tour, country, neut, targets):
        preds = self.forward(t1, t2, t1_adv, t1_res, t2_adv, t2_res, tour, country, neut)
        return self.loss(preds, targets.squeeze(1))

class CrossNetwork(nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x