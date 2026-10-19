from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .common import find_csvs, unique_files_by_content, normalise_label
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def load_sample(files, max_per_class, chunksize=150000, seed=42):
    pieces=[]
    # Keep the intermediate sample bounded when scanning many millions of rows.
    chunk_quota = min(max_per_class, max(100, max_per_class // 10))
    for path in files:
        print(f"[SAMPLE] {path.name}")
        for chunk in pd.read_csv(path, usecols=lambda c: c in ["ts","src_port","dst_port","proto","type"],
                                 chunksize=chunksize, low_memory=False):
            if "type" not in chunk: continue
            chunk["type"]=normalise_label(chunk["type"])
            chunk=chunk.dropna(subset=["type"])
            for _, group in chunk.groupby("type", sort=False):
                n=min(len(group), chunk_quota)
                pieces.append(group.sample(n=n, random_state=seed) if len(group)>n else group)
    if not pieces: raise RuntimeError("No network rows loaded; check input_data/network.")
    df=pd.concat(pieces, ignore_index=True).drop_duplicates()
    final=[]
    for _,g in df.groupby("type", sort=False):
        final.append(g.sample(n=min(len(g),max_per_class),random_state=seed))
    return pd.concat(final,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)

class WindowDataset(Dataset):
    def __init__(self, X, y, window, sample_ids=None):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
        self.window=window
        self.sample_ids=np.asarray(sample_ids if sample_ids is not None else np.arange(len(y)))
    def __len__(self): return len(self.X)-self.window+1
    def __getitem__(self, idx):
        # The target is the final row. See the output warning: sampled rows are not guaranteed to be contiguous traffic.
        return self.X[idx:idx+self.window], self.y[idx+self.window-1]

class CNNModel(nn.Module):
    def __init__(self,n_features,n_classes):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(n_features,32,3,padding=1),nn.ReLU(),
            nn.Conv1d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),
            nn.Linear(32,n_classes))
    def forward(self,x): return self.net(x.transpose(1,2))

class LSTMModel(nn.Module):
    def __init__(self,n_features,n_classes):
        super().__init__(); self.rnn=nn.LSTM(n_features,32,batch_first=True)
        self.fc=nn.Linear(32,n_classes)
    def forward(self,x):
        _,(h,_)=self.rnn(x); return self.fc(h[-1])

class TransformerModel(nn.Module):
    def __init__(self,n_features,n_classes,window):
        super().__init__()
        self.proj=nn.Linear(n_features,32)
        layer=nn.TransformerEncoderLayer(d_model=32,nhead=4,dim_feedforward=64,batch_first=True,dropout=0.1)
        self.enc=nn.TransformerEncoder(layer,num_layers=1)
        self.fc=nn.Linear(32,n_classes)
    def forward(self,x): return self.fc(self.enc(self.proj(x))[:,-1,:])

def run_model(name, model, train_loader, test_loader, device, epochs):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); total=0
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb)
        print(f"{name} epoch {ep+1}/{epochs}: loss={total/max(1,len(train_loader.dataset)):.4f}")
    model.eval(); preds=[]; actual=[]
    with torch.no_grad():
        for xb,yb in test_loader:
            logits=model(xb.to(device)); preds.extend(logits.argmax(1).cpu().numpy()); actual.extend(yb.numpy())
    return {"accuracy":float(accuracy_score(actual,preds)),
            "macro_f1":float(f1_score(actual,preds,average="macro",zero_division=0)),
            "classification_report":classification_report(actual,preds,output_dict=True,zero_division=0)}, model

def main():
    p=argparse.ArgumentParser(description="Compact DL feasibility experiment (not a validated temporal IDS).")
    p.add_argument("--network-dir",default="input_data/network")
    p.add_argument("--max-per-class",type=int,default=3000)
    p.add_argument("--window",type=int,default=8)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    out=Path("reports/dl"); out.mkdir(parents=True,exist_ok=True); Path("models").mkdir(exist_ok=True)
    files,dups=unique_files_by_content(find_csvs(args.network_dir))
    if not files: raise FileNotFoundError(f"No network CSVs under {args.network_dir}")
    shared_sample = Path("reports/modeling_sample.csv")
    if shared_sample.exists():
        print(f"[SAMPLE] Using shared traditional/DL sample: {shared_sample}")
        df = pd.read_csv(shared_sample, low_memory=False)
        df["type"] = normalise_label(df["type"])
        if "sample_id" not in df.columns:
            df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    else:
        print("[WARN] Shared modeling sample not found. Run run_baseline.bat first for matched pilot IDs.")
        df=load_sample(files,args.max_per_class,seed=args.seed)
        df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    labels=normalise_label(df["type"]).astype(str)
    enc=LabelEncoder(); y=enc.fit_transform(labels)
    proto=pd.get_dummies(df["proto"].astype("string").fillna("unknown").str.lower(),prefix="proto",dtype=np.uint8)
    raw=pd.DataFrame({"src_port":pd.to_numeric(df["src_port"],errors="coerce").fillna(-1).clip(-1,65535),
                      "dst_port":pd.to_numeric(df["dst_port"],errors="coerce").fillna(-1).clip(-1,65535)})
    X=pd.concat([raw.reset_index(drop=True),proto.reset_index(drop=True)],axis=1).astype(np.float32).to_numpy()
    # Same stratified indices/seed as the traditional trainer. Fit scaling on training rows only.
    idx=np.arange(len(y)); train_idx,test_idx=train_test_split(idx,test_size=0.2,random_state=args.seed,stratify=y)
    scaler=StandardScaler(); scaler.fit(X[train_idx]); X=scaler.transform(X).astype(np.float32)
    sample_ids=df["sample_id"].to_numpy()
    # This is still a feasibility sequence: the sampled points are not guaranteed to be adjacent packets.
    def make_windows(ix):
        xx=X[ix]; yy=y[ix]; ids=sample_ids[ix]
        if len(ix)<args.window: raise RuntimeError("Not enough rows for requested window size.")
        return WindowDataset(xx,yy,args.window,ids)
    train_ds=make_windows(train_idx); test_ds=make_windows(test_idx)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False)
    n_features=X.shape[1]; n_classes=len(enc.classes_)
    device="cuda" if torch.cuda.is_available() else "cpu"
    models={"cnn":CNNModel(n_features,n_classes),"lstm":LSTMModel(n_features,n_classes),
            "transformer":TransformerModel(n_features,n_classes,args.window)}
    results={}
    for name,model in models.items():
        print(f"\nTraining {name} on {device}...")
        metrics,fitted=run_model(name,model,train_loader,test_loader,device,args.epochs)
        results[name]={"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"],
                        "classification_report":metrics["classification_report"]}
        torch.save({"state_dict":fitted.state_dict(),"classes":enc.classes_.tolist(),
                    "n_features":n_features,"window":args.window,"model":name,
                    "feature_columns":["src_port","dst_port"]+list(proto.columns),
                    "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()},
                   Path("models")/f"{name}_feasibility.pt")
        if name == "transformer":
            # Export predictions tied to the final row of each sequence for the shared pilot.
            fitted.eval(); dl_pred=[]; dl_true=[]
            with torch.no_grad():
                for xb,yb in test_loader:
                    logits=fitted(xb.to(device)); dl_pred.extend(logits.argmax(1).cpu().numpy()); dl_true.extend(yb.numpy())
            end_ids=test_ds.sample_ids[args.window-1:]
            pd.DataFrame({"sample_id":end_ids[:len(dl_pred)],
                          "dl_true_label":enc.inverse_transform(np.asarray(dl_true,dtype=int)),
                          "transformer_prediction":enc.inverse_transform(np.asarray(dl_pred,dtype=int)),
                          "transformer_correct":np.asarray(dl_true)==np.asarray(dl_pred)}).to_csv(
                              out/"transformer_test_predictions.csv",index=False)
            # Save a small held-out window array for the pilot/explanation stage.
            np.savez_compressed(out/"transformer_test_windows.npz",
                                X=np.stack([test_ds[i][0].numpy() for i in range(len(test_ds))]),
                                y=np.asarray([test_ds[i][1].item() for i in range(len(test_ds))]),
                                sample_id=end_ids)
            try:
                import shap
                bg_n=min(40,len(train_ds)); ex_n=min(80,len(test_ds))
                background=torch.stack([train_ds[i][0] for i in range(bg_n)]).to(device)
                explain=torch.stack([test_ds[i][0] for i in range(ex_n)]).to(device)
                explainer=shap.GradientExplainer(fitted,background)
                shap_values=explainer.shap_values(explain)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                # Aggregate absolute attribution across windows and classes for a readable global summary.
                if isinstance(shap_values,list):
                    arr=np.mean([np.abs(np.asarray(v)) for v in shap_values],axis=0)
                else:
                    arr=np.abs(np.asarray(shap_values))
                    if arr.ndim==4: arr=arr.mean(axis=-1)
                if arr.ndim==3: arr=arr.mean(axis=1)
                mean_abs=arr.mean(axis=0).reshape(-1)
                feat_names=["src_port","dst_port"]+list(proto.columns)
                pd.DataFrame({"feature":feat_names,"mean_abs_shap":mean_abs[:len(feat_names)]}).sort_values(
                    "mean_abs_shap",ascending=False).to_csv(out/"transformer_shap_importance.csv",index=False)
                plt.figure(figsize=(8,4)); imp=pd.DataFrame({"feature":feat_names,"mean_abs_shap":mean_abs[:len(feat_names)]}).sort_values("mean_abs_shap")
                plt.barh(imp["feature"],imp["mean_abs_shap"]); plt.xlabel("Mean absolute SHAP value"); plt.tight_layout()
                plt.savefig(out/"transformer_shap_summary.png",dpi=150); plt.close()
            except Exception as exc:
                print(f"[WARN] DL SHAP skipped: {exc}")
    payload={"device":device,"rows_sampled":len(df),"classes":enc.classes_.tolist(),
             "max_per_class":args.max_per_class,"window":args.window,"epochs":args.epochs,
             "warning":"Feasibility only: windows are formed from sampled rows and are not guaranteed to be contiguous traffic. The stratified random split is not a real-world generalisation test.",
             "shared_sample": str(shared_sample) if shared_sample.exists() else None,
             "pilot_predictions": "reports/dl/transformer_test_predictions.csv",
             "dl_shap": "reports/dl/transformer_shap_summary.png (if SHAP GradientExplainer succeeds)",
             "results":results}
    (out/"dl_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:{"accuracy":v["accuracy"],"macro_f1":v["macro_f1"]} for k,v in results.items()},indent=2))
    print("DL feasibility results saved to reports/dl/dl_metrics.json")

if __name__=="__main__": main()
