"""
train_model.py
==============

一支「先能跑再優化」的最小 PyTorch 訓練腳本，用我們
LoggingAgent 產生的 JSONL 做 Imitation Learning。

python ./train/train_model.py --log-dir logs --epochs 50 --device cpu
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Dict, Tuple

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# 基本卡牌編碼工具
# ----------------------------------------------------------------------
SUITS = "CDHS"                       # C♣ D♦ H♥ S♠
RANKS = "23456789TJQKA"
CARD2IDX = {r + s: 13 * si + ri      # 0–51
            for si, s in enumerate(SUITS)
            for ri, r in enumerate(RANKS)}
PAD_IDX = 52                         # 52 當 padding

def encode_card(card: str) -> int:
    """'C9' → 33；空字串回傳 PAD_IDX"""
    return CARD2IDX.get(card, PAD_IDX)

def pad(seq: List[int], size: int, val: int = PAD_IDX) -> List[int]:
    return (seq + [val] * size)[:size]

# ----------------------------------------------------------------------
# Dataset：把 declare_action 事件轉成 (觀測, 標籤)
# ----------------------------------------------------------------------
class PokerDataset(Dataset):
    def __init__(self, log_dir: str | Path,
                 bucket: Dict[str, int] | None = None):
        self.labels = bucket or {"fold": 0, "call": 1, "raise": 2}
        self.data: List[Tuple] = []  # (hole[2], board[5], num_feat[4], y)

        for path in Path(log_dir).glob("*.jsonl"):
            with path.open() as fp:
                for line in fp:
                    evt = json.loads(line)
                    if evt["type"] != "declare_action":
                        continue
                    # 卡牌
                    hole = [encode_card(c) for c in evt["hole_card"]]
                    board = pad([encode_card(c) for c in
                                 evt["round_state"]["community_card"]], 5)
                    # 數值特徵 (可自行增減)
                    rs = evt["round_state"]
                    pot = rs["pot"]["main"]["amount"] / 2000           # 粗略 normalise
                    sb  = rs["small_blind_amount"] / 10
                    eq = act["equity"]                  # 讀新欄位
                    street_id = ["preflop", "flop", "turn",
                                 "river", "showdown"].index(rs["street"]) / 4
                    to_call = next((a["amount"] for a in evt["valid_actions"]
                                    if a["action"] == "call"), 0) / 500
                    num_feat = torch.tensor([pot, sb, street_id, to_call, eq],
                                            dtype=torch.float32)
                    # 標籤
                    action, _ = evt["chosen"]
                    y = self.labels[action]
                    self.data.append(((hole, board, num_feat), y))

        random.shuffle(self.data)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        (hole, board, f), y = self.data[idx]
        return (torch.tensor(hole, dtype=torch.long),
                torch.tensor(board, dtype=torch.long),
                f,
                torch.tensor(y, dtype=torch.long))

# ----------------------------------------------------------------------
# 簡易網路：牌嵌入 + 數值 → MLP
# ----------------------------------------------------------------------
class SimplePolicyNet(nn.Module):
    def __init__(self, n_action: int = 3,
                 emb_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(53, emb_dim, padding_idx=PAD_IDX)
        in_dim = emb_dim * 7 + 4    # 2 hole + 5 board + num_feat
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden *4), 
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(hidden *4, hidden*2),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden *2, hidden*2),
            nn.Dropout(0.0),
            nn.Sigmoid(),
            nn.Linear(hidden *2, hidden),
            nn.Dropout(0.0),
            nn.GELU(),
            nn.Linear(hidden, n_action)
        )

    def forward(self, hole, board, feat):
        x = torch.cat([self.emb(hole).view(hole.size(0), -1),
                       self.emb(board).view(board.size(0), -1),
                       feat], dim=-1)
        return self.mlp(x)

# ----------------------------------------------------------------------
# 訓練 / 驗證
# ----------------------------------------------------------------------
def epoch_run(model, loader, crit, opt=None, device="cpu"):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for hole, board, feat, y in loader:
        hole, board, feat, y = (t.to(device) for t in (hole, board, feat, y))
        logits = model(hole, board, feat)
        loss = crit(logits, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

# ----------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    ds = PokerDataset(args.log_dir)
    val_sz = int(len(ds) * 0.1)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - val_sz, val_sz])

    dl_train = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    dl_val   = DataLoader(val_ds,   batch_size=args.batch)

    model = SimplePolicyNet().to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = epoch_run(model, dl_train, crit, opt, device)
        vl_loss, vl_acc = epoch_run(model, dl_val,   crit, None, device)
        print(f"[{epoch:02d}] "
              f"train {tr_loss:.4f}/{tr_acc:.3f}  "
              f"val {vl_loss:.4f}/{vl_acc:.3f}")

    torch.jit.script(model.cpu()).save(args.out)
    print("model saved →", args.out)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="poker_policy.pt")
    return p.parse_args()

if __name__ == "__main__":
    main()