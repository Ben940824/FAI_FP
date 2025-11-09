"""
train_model_rnn.py
==================

加入「最近 3 筆行動序列 → GRU」的版本，能學到同一 street
連續下注模式。執行範例：

    python ./train/train_model_rnn.py --log-dir logs --epochs 100 --device cpu
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Tuple, Dict

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------- 基本常數
SUITS = "CDHS"
RANKS = "23456789TJQKA"
CARD2IDX = {r + s: 13 * si + ri for si, s in enumerate(SUITS) for ri, r in enumerate(RANKS)}
PAD_CARD = 52  # card-embedding padding idx

ACTION_VOCAB = {
    "FOLD": 0, "CALL": 1, "CHECK": 2, "RAISE": 3,
    "SMALLBLIND": 4, "BIGBLIND": 5, "UNKNOWN": 6,
}
PAD_ACT = len(ACTION_VOCAB)          # action-seq padding
SEQ_LEN = 3                          # 取最近 3 步

# ------------------------------------------------- encode helpers
def enc_card(c: str) -> int: return CARD2IDX.get(c, PAD_CARD)
def pad(lst: List[int], n: int, val: int) -> List[int]: return (lst + [val]*n)[:n]

# ------------------------------------------------- Dataset
class PokerDataset(Dataset):
    def __init__(self, log_dir: str | Path, bucket: Dict[str,int]|None=None):
        self.label_map = bucket or {"fold":0,"call":1,"raise":2}
        self.samples: List[Tuple] = []

        for p in Path(log_dir).glob("*.jsonl"):
            for line in p.open():
                e = json.loads(line)
                if e["type"] != "declare_action": continue

                rs = e["round_state"]
                # ---- action history （整局展開後取最後 3 步）----
                hist = []
                for st in ("preflop","flop","turn","river"):
                    hist.extend(rs["action_histories"].get(st, []))
                act_seq = [ACTION_VOCAB.get(a["action"], ACTION_VOCAB["UNKNOWN"])
                           for a in hist[-SEQ_LEN:]]
                act_seq = pad(act_seq, SEQ_LEN, PAD_ACT)

                # ---- cards & numeric feats ----
                hole  = [enc_card(c) for c in e["hole_card"]]
                board = pad([enc_card(c) for c in rs["community_card"]], 5, PAD_CARD)
                pot   = rs["pot"]["main"]["amount"] / 2000
                sb    = rs["small_blind_amount"] / 10
                street_id = ["preflop","flop","turn","river","showdown"].index(rs["street"]) / 4
                to_call = next((a["amount"] for a in e["valid_actions"] if a["action"]=="call"),0)/500
                num_feat = torch.tensor([pot,sb,street_id,to_call], dtype=torch.float32)

                y = self.label_map[e["chosen"][0]]
                self.samples.append(((hole,board,num_feat,act_seq), y))

        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i: int):
        (h,b,f,a),y = self.samples[i]
        return (torch.tensor(h), torch.tensor(b), f,
                torch.tensor(a), torch.tensor(y))

# ------------------------------------------------- Model
class RNNPolicy(nn.Module):
    def __init__(self, n_out=3, card_emb=16, act_emb=8, hid=128):
        super().__init__()
        self.card_emb = nn.Embedding(53, card_emb, padding_idx=PAD_CARD)
        self.act_emb  = nn.Embedding(len(ACTION_VOCAB)+1, act_emb, padding_idx=PAD_ACT)
        self.gru = nn.GRU(act_emb, 32, batch_first=True)
        in_dim = card_emb*7 + 32 + 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), 
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, n_out)
        )
    def forward(self, h, b, f, a):
        h_feat = self.card_emb(h).view(h.size(0), -1)
        b_feat = self.card_emb(b).view(b.size(0), -1)
        _, g = self.gru(self.act_emb(a))
        x = torch.cat([h_feat, b_feat, g.squeeze(0), f], dim=-1)
        return self.mlp(x)

# ------------------------------------------------- Train utils
def run(model, loader, crit, opt=None, dev="cpu"):
    train = opt is not None; model.train(train)
    tot=acc=loss=0.0
    for h,b,f,a,y in loader:
        h,b,f,a,y = (t.to(dev) for t in (h,b,f,a,y))
        logit = model(h,b,f,a); l = crit(logit,y)
        if train: opt.zero_grad(); l.backward(); opt.step()
        loss += l.item()*y.size(0); acc += (logit.argmax(1)==y).sum().item(); tot+=y.size(0)
    return loss/tot, acc/tot

# ------------------------------------------------- main
def main():
    args = get_args(); dev = torch.device(args.device)
    ds = PokerDataset(args.log_dir); val = int(len(ds)*.1)
    tr,va = torch.utils.data.random_split(ds,[len(ds)-val,val])
    dl_tr = DataLoader(tr,batch_size=args.batch,shuffle=True)
    dl_va = DataLoader(va,batch_size=args.batch)

    net = RNNPolicy().to(dev); crit=nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    for ep in range(1,args.epochs+1):
        tr_l,tr_a = run(net,dl_tr,crit,opt,dev)
        va_l,va_a = run(net,dl_va,crit,None,dev)
        print(f"[{ep:02d}] train {tr_l:.4f}/{tr_a:.3f} | val {va_l:.4f}/{va_a:.3f}")

    torch.jit.script(net.cpu()).save(args.out); print("✓ saved →", args.out)

def get_args():
    p=argparse.ArgumentParser()
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="poker_policy_rnn.pt")
    return p.parse_args()

if __name__=="__main__": main()