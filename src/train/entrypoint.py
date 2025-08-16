import os, json, random, numpy as np, torch, torch.nn as nn, tarfile, glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from src.models.audio_cnn import AudioCNN
from src.utils.audio import wav_to_melspec, normalize_spec
import librosa

CLASSES = ["air_conditioner","car_horn","children_playing","dog_bark","drilling",
           "engine_idling","gun_shot","jackhammer","siren","street_music"]

DATA_CHANNEL = Path("/opt/ml/input/data/train")

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def prepare_dataset_path() -> Path:
    """
    If a .tar.gz is present in the channel, extract it to /opt/ml/input/data/train/extracted
    and return that path; else return the channel as-is.
    """
    DATA_CHANNEL.mkdir(parents=True, exist_ok=True)
    tars = list(DATA_CHANNEL.glob("*.tar.gz")) + list(DATA_CHANNEL.glob("*.tgz"))
    if tars:
        tar_path = tars[0]
        extract_dir = DATA_CHANNEL / "extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(path=extract_dir)
        return extract_dir
    return DATA_CHANNEL

class UrbanSoundFolderDataset(Dataset):
    """
    Walk UrbanSound tree. Your archive shows: UrbanSound/data/<class>/*.mp3
    """
    def __init__(self, root: Path, n_mels=64, sr=22050, dur=2.0):
        self.n_mels = n_mels
        self.sr = sr
        self.dur = dur
        self.items = []
        # root may contain UrbanSound/, or directly data/
        if (root/"UrbanSound"/"data").exists():
            data_root = root/"UrbanSound"/"data"
        elif (root/"data").exists():
            data_root = root/"data"
        else:
            data_root = root
        # class folders
        class_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
        self.class_names = [p.name for p in class_dirs]
        self.class_to_id = {c:i for i,c in enumerate(self.class_names)}
        # files
        for cls in self.class_names:
            # mp3/wav
            for fp in glob.glob(str(data_root/cls/"*.mp3")) + glob.glob(str(data_root/cls/"*.wav")):
                self.items.append((fp, self.class_to_id[cls]))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fp, y = self.items[idx]
        x, sr = librosa.load(fp, sr=self.sr, mono=True, duration=self.dur)
        spec = normalize_spec(wav_to_melspec(x, sr, n_mels=self.n_mels))
        spec = torch.from_numpy(spec).unsqueeze(0)  # (1,n_mels,T)
        return spec, torch.tensor(y, dtype=torch.long)

class SyntheticToneDataset(Dataset):
    def __init__(self, n=800, sr=22050, dur=2.0, n_mels=64):
        self.items=[]
        n_samples=int(sr*dur)
        t=np.linspace(0,dur,n_samples,endpoint=False)
        for _ in range(n):
            f=np.random.uniform(100,4000)
            y=0.5*np.sin(2*np.pi*f*t).astype("float32")+0.05*np.random.randn(n_samples).astype("float32")
            spec=normalize_spec(wav_to_melspec(y, sr, n_fft=1024, hop_length=256, n_mels=n_mels))
            label=np.random.randint(0,10)
            self.items.append((spec,label))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        spec,label=self.items[i]
        x=torch.from_numpy(spec).unsqueeze(0)
        return x, torch.tensor(label, dtype=torch.long)

def train():
    seed_all(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_mels, batch, epochs = 64, 64, 8

    data_root = prepare_dataset_path()
    real_files = list(data_root.rglob("*.mp3")) + list(data_root.rglob("*.wav"))
    if real_files:
        print(f"Found {len(real_files)} audio files under {data_root} – using real dataset.")
        ds = UrbanSoundFolderDataset(root=data_root, n_mels=n_mels)
        class_names = ds.class_names if hasattr(ds, "class_names") else CLASSES
    else:
        print("No real data found – using synthetic dataset (CI mode).")
        ds = SyntheticToneDataset(n=800, n_mels=n_mels)
        class_names = CLASSES

    tr_len = int(0.8*len(ds)); va_len = max(1, len(ds)-tr_len)
    tr,va = random_split(ds,[tr_len,va_len], generator=torch.Generator().manual_seed(42))
    tr_loader=DataLoader(tr,batch_size=batch,shuffle=True,num_workers=2)
    va_loader=DataLoader(va,batch_size=batch,shuffle=False,num_workers=2)

    model=AudioCNN(n_mels=n_mels,num_classes=len(class_names)).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    crit=nn.CrossEntropyLoss()
    best=-1.0
    for e in range(1,epochs+1):
        model.train(); tl=0; ta=0; n=0
        for x,y in tr_loader:
            x,y=x.to(device), y.to(device); opt.zero_grad()
            logits=model(x); loss=crit(logits,y); loss.backward(); opt.step()
            tl+=float(loss)*y.size(0); ta+=(logits.argmax(1)==y).sum().item(); n+=y.size(0)
        model.eval(); vl=0; vaa=0; m=0
        with torch.no_grad():
            for x,y in va_loader:
                x,y=x.to(device), y.to(device)
                logits=model(x); loss=crit(logits,y)
                vl+=float(loss)*y.size(0); vaa+=(logits.argmax(1)==y).sum().item(); m+=y.size(0)
        print(f"[{e}/{epochs}] train_loss={tl/n:.4f} acc={ta/n:.3f} | val_loss={vl/m:.4f} acc={vaa/m:.3f}")
        if m>0 and vaa/m>best:
            best=vaa/m
            out=Path("/opt/ml/model"); out.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict":model.state_dict(),"n_mels":n_mels,"num_classes":len(class_names)}, out/"model.pt")
            (out/"classes.json").write_text(json.dumps(class_names))
    print("✅ best_val_acc:", best)

if __name__=="__main__":
    train()

