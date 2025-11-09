"""
Imitation + Value (AWAC-lite)
-----------------------------
  • policy_head: CrossEntropy(logits → {fold,call,raise})
  • value_head : Regression to reward_norm  (∈[-1,+1])
  • policy loss weighted by exp(beta * advantage)

  python ./train/train_model_awac.py --log-dir logs --epochs 200 --device cpu

"""

from __future__ import annotations
import argparse, json, random, math
from pathlib import Path
from typing import List, Tuple, Dict

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 基本常數 ----------
SUITS,RANKS="CDHS","23456789TJQKA"
CARD2IDX={r+s:13*si+ri for si,s in enumerate(SUITS) for ri,r in enumerate(RANKS)}
PAD_CARD=52; INITIAL_STACK=1000

def enc(c:str)->int: return CARD2IDX.get(c, PAD_CARD)
def pad(seq,k,val): return (seq+[val]*k)[:k]

# ---------- Dataset ----------
class PokerDS(Dataset):
    def __init__(self, log_dir, bucket=None):
        bucket = bucket or {"fold":0,"call":1,"raise":2}
        self.samples = []

        for p in Path(log_dir).glob("*.jsonl"):
            reward_of_round: Dict[int,float] = {}
            decl_by_round: Dict[int,List[dict]] = {}

            for ln in p.open():
                e = json.loads(ln)

                # --------- 收 reward ---------
                if e["type"] == "round_result":
                    rc   = e["round_count"]
                    # 老師永遠是座位列表第一個 (因為你只 log 老師的 declare_action)
                    teacher_stack = e["seats"][0]["stack"]
                    reward_of_round[rc] = (teacher_stack - INITIAL_STACK) / INITIAL_STACK
                    continue

                # --------- 收 declare_action ---------
                if e["type"] == "declare_action":
                    rc = e["round_count"]
                    decl_by_round.setdefault(rc, []).append(e)

            # 建立樣本：同一 round 的多步 declare_action 共用一個 reward
            for rc, acts in decl_by_round.items():
                if rc not in reward_of_round:       # safety
                    continue
                rwd = reward_of_round[rc]
                for act in acts:
                    rs  = act["round_state"]
                    hole  = [enc(c) for c in act["hole_card"]]
                    board = pad([enc(c) for c in rs["community_card"]], 5, PAD_CARD)
                    # ---- numeric feats ----
                    pot = rs["pot"]["main"]["amount"] / 2000
                    sb  = rs["small_blind_amount"] / 10
                    street_id = ["preflop","flop","turn","river","showdown"].index(rs["street"]) / 4
                    to_call = next(a["amount"] for a in act["valid_actions"] if a["action"] == "call") / 500
                    num = torch.tensor([pot, sb, street_id, to_call], dtype=torch.float32)

                    y = bucket[act["chosen"][0]]
                    self.samples.append(((hole, board, num), y, rwd))

        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)
    def __getitem__(self,i:int):
        (h,b,f),y,r=self.samples[i]
        return (torch.tensor(h),torch.tensor(b),f,
                torch.tensor(y),torch.tensor(r,dtype=torch.float32))

# ---------- Model ----------
class Net(nn.Module):
    def __init__(self,n_act=3,emb=16,hid=128):
        super().__init__()
        self.emb=nn.Embedding(53,emb,padding_idx=PAD_CARD)
        in_dim=emb*7+4
        self.mlp=nn.Sequential(nn.Linear(in_dim,hid),nn.ReLU())
        self.policy=nn.Linear(hid,n_act)
        self.value =nn.Linear(hid,1)
    def forward(self,h,b,f):
        x=torch.cat([self.emb(h).view(h.size(0),-1),
                     self.emb(b).view(b.size(0),-1),f],dim=-1)
        h=self.mlp(x)
        return self.policy(h), self.value(h).squeeze(-1)  # logits, v

# ---------- Train Loop ----------
def run_epoch(net,ld,crit_ce,crit_mse,opt=None,beta=5.0,dev="cpu"):
    train=opt is not None; net.train(train)
    tot=acc=ls=0.0
    for h,b,f,y,r in ld:
        h,b,f,y,r=(t.to(dev) for t in (h,b,f,y,r))
        logits,v=net(h,b,f)
        ce=crit_ce(logits,y)
        with torch.no_grad():
            adv=(r - v).clamp(-1,1)
            w = torch.exp(beta*adv)
        loss_policy=(ce*w).mean()
        loss_value=crit_mse(v,r)
        loss=loss_policy+0.5*loss_value
        if train: opt.zero_grad(); loss.backward(); opt.step()
        ls+=loss.item()*y.size(0); acc+=(logits.argmax(1)==y).sum().item(); tot+=y.size(0)
    return ls/tot, acc/tot

def main():
    args=parse(); dev=torch.device(args.device)
    ds=PokerDS(args.log_dir); val=int(len(ds)*.1)
    tr,va=torch.utils.data.random_split(ds,[len(ds)-val,val])
    dl_tr=DataLoader(tr,batch_size=args.batch,shuffle=True)
    dl_va=DataLoader(va,batch_size=args.batch)
    net=Net().to(dev)
    ce=nn.CrossEntropyLoss()
    mse=nn.MSELoss()
    opt=torch.optim.AdamW(net.parameters(), lr=args.lr)
    for ep in range(1,args.epochs+1):
        l,a=run_epoch(net,dl_tr,ce,mse,opt,dev=dev)
        vl,va=run_epoch(net,dl_va,ce,mse,None,dev=dev)
        print(f"[{ep:02d}] train {l:.4f}/{a:.3f} | val {vl:.4f}/{va:.3f}")
    torch.jit.script(net.cpu()).save(args.out); print("✓ saved →",args.out)

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--log-dir",default="logs"); p.add_argument("--batch",type=int,default=256)
    p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--device",default="cpu"); p.add_argument("--out",default="poker_policy_awac.pt")
    return p.parse_args()

if __name__=="__main__": main()