
import argparse, json, os, re, random
from collections import Counter
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PAD, UNK, SEP = 0, 1, 2
dev = "cuda" if torch.cuda.is_available() else "cpu"

def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def tok(t):
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", str(t).lower())

def build_vocab(texts, max_size=20000):
    c = Counter(w for t in texts for w in tok(t))
    return {w: i + 3 for i, (w, _) in enumerate(c.most_common(max_size))}

def encode(inp, out, vocab, mode, li=64, lo=256):
    def ids(t, n):
        return [vocab.get(w, UNK) for w in tok(t)][:n]
    if mode == "output": s = ids(out, lo)
    elif mode == "input": s = ids(inp, li)
    else: s = ids(inp, li) + [SEP] + ids(out, lo)
    return s

def pad_batch(seqs):
    n = max(1, max(len(s) for s in seqs))
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs])

class TextCNN(nn.Module):
    def __init__(self, V, C, d=128, nf=100, ks=(3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(d, nf, k, padding=k // 2) for k in ks])
        self.drop = nn.Dropout(0.5); self.fc = nn.Linear(nf * len(ks), C)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).max(2).values for c in self.convs], 1)
        return self.fc(self.drop(h))

class BiLSTM(nn.Module):
    def __init__(self, V, C, d=128, h=128):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.rnn = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5); self.fc = nn.Linear(2 * h, C)
    def forward(self, x):
        o, _ = self.rnn(self.emb(x))
        mask = (x != PAD).unsqueeze(-1)
        h = (o * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean pool
        return self.fc(self.drop(h))

def batches(X, y, bs, shuffle):
    idx = list(range(len(X)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield pad_batch([X[j] for j in b]).to(dev), torch.tensor([y[j] for j in b]).to(dev)

def predict(m, X, y, bs=128):
    m.eval(); P = []
    with torch.no_grad():
        for xb, _ in batches(X, y, bs, False):
            P += m(xb).argmax(1).tolist()
    return P

def train_eval(kind, Xtr, ytr, Xva, yva, Xte, yte, V, C, epochs=12, lr=1e-3):
    m = (TextCNN if kind == "cnn" else BiLSTM)(V, C).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss(); best, state, hist = -1, None, []
    for ep in range(epochs):
        m.train(); tot = 0
        for xb, yb in batches(Xtr, ytr, 64, True):
            opt.zero_grad(); l = lossf(m(xb), yb); l.backward(); opt.step(); tot += l.item()
        va = accuracy_score(yva, predict(m, Xva, yva))
        hist.append({"epoch": ep + 1, "train_loss": tot, "val_acc": va})
        if va > best: best, state = va, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    p = predict(m, Xte, yte)
    return {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"),
            "cm": confusion_matrix(yte, p).tolist(), "history": hist}

def run_setting(tr, va, te, mode, kinds, C, transform=None):
    f = (lambda d: d) if transform is None else transform
    tr, va, te = f(tr), f(va), f(te)
    vocab = build_vocab((tr.input + " " + tr.output) if mode != "output" else tr.output)
    enc = lambda d: [encode(a, b, vocab, mode) for a, b in zip(d.input, d.output)]
    Xtr, Xva, Xte = enc(tr), enc(va), enc(te)
    res = {}
    for k in kinds:
        seed_all(0)
        res[k] = train_eval(k, Xtr, list(tr.y), Xva, list(va.y), Xte, list(te.y), len(vocab) + 3, C)
        print(f"  {mode:6s} {k:4s} acc={res[k]['acc']:.3f} f1={res[k]['macro_f1']:.3f}")
    return res

# ---------- RQ4 helpers ----------
def feats(t):
    t = str(t); w = t.split(); n = max(1, len(w))
    return [len(t), len(w), len(set(x.lower() for x in w)) / n, t.count("\n"),
            t.count("**"), t.count("#"), len(re.findall(r"^\s*[-*] ", t, re.M)),
            len(re.findall(r"^\s*\d+\. ", t, re.M)), t.count("```"), t.count(","),
            t.count("!"), sum(c.isdigit() for c in t) / max(1, len(t)), np.mean([len(x) for x in w]) if w else 0]
FEAT_NAMES = ["chars", "words", "type_token_ratio", "newlines", "bold_marks", "headers",
              "bullets", "numbered", "code_fences", "commas", "exclaims", "digit_frac", "avg_word_len"]

def strip_format(d):
    d = d.copy()
    def clean(t):
        t = re.sub(r"```.*?```", " code ", str(t), flags=re.S)
        t = re.sub(r"[*#`>_\-]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()
    d["output"] = d.output.map(clean); return d

def truncate_len(d, n=60):
    d = d.copy(); d["output"] = d.output.map(lambda t: " ".join(str(t).split()[:n])); return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--per_class", type=int, default=0, help="cap examples per LLM (0 = min class size)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); seed_all(0)

    df = pd.read_csv(a.data).dropna(subset=["llm", "input", "output"])
    cap = a.per_class or df.llm.value_counts().min()
    df = pd.concat([g.sample(min(len(g), cap), random_state=0) for _, g in df.groupby("llm")])
    classes = sorted(df.llm.unique()); df["y"] = df.llm.map({c: i for i, c in enumerate(classes)})
    C = len(classes); print("classes:", classes, df.llm.value_counts().to_dict())
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df.y, random_state=0)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.y, random_state=0)
    R = {"classes": classes, "sizes": [len(tr), len(va), len(te)]}

    print("RQ1 + RQ2"); R["rq1_rq2"] = {m: run_setting(tr, va, te, m, ["cnn", "lstm"], C)
                                         for m in ["output", "input", "both"]}

    if "task" in df.columns:
        print("RQ3 leave-one-task-out"); R["rq3"] = {}
        for t in sorted(df.task.dropna().unique()):
            trn, tst = df[df.task != t], df[df.task == t]
            if len(tst) < 30: continue
            trn_tr, trn_va = train_test_split(trn, test_size=0.1, stratify=trn.y, random_state=0)
            print(" held out:", t)
            R["rq3"][t] = run_setting(trn_tr, trn_va, tst, "output", ["cnn", "lstm"], C)

    print("RQ4"); r4 = {}
    r4["stats"] = {c: dict(zip(FEAT_NAMES, np.mean([feats(x) for x in g.output], 0).round(3).tolist()))
                   for c, g in df.groupby("llm")}
    sc = StandardScaler().fit([feats(x) for x in tr.output])
    lr = LogisticRegression(max_iter=2000).fit(sc.transform([feats(x) for x in tr.output]), tr.y)
    p = lr.predict(sc.transform([feats(x) for x in te.output]))
    r4["handcrafted_logreg"] = {"acc": accuracy_score(te.y, p), "macro_f1": f1_score(te.y, p, average="macro")}
    r4["top_features"] = {c: [FEAT_NAMES[i] for i in np.argsort(-np.abs(lr.coef_[k]))[:3]] for k, c in enumerate(classes)}
    print(" ablation: strip formatting"); r4["no_format"] = run_setting(tr, va, te, "output", ["cnn", "lstm"], C, strip_format)
    print(" ablation: truncate to 60 words"); r4["truncated"] = run_setting(tr, va, te, "output", ["cnn", "lstm"], C, truncate_len)
    R["rq4"] = r4

    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    print("saved", os.path.join(a.out, "results.json"))

if __name__ == "__main__":
    main()
