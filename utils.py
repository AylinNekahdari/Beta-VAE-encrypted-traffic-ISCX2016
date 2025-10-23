import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import torch

def load_and_preprocess(data_path, label_col="traffic_type", seed=42):
    df = pd.read_csv(data_path)

    # Use 'traffic_type' as the label column
    label_col = "traffic_type"  
    df[label_col] = df[label_col].astype(str).str.strip()

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[label_col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "label_enc"]

    X = df[feature_cols].fillna(0).values
    y = df["label_enc"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.17647, stratify=y_trainval, random_state=seed
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test, scaler, le)


def extract_latents(model, loader, device):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb_f = xb.to(device), yb.to(device).unsqueeze(1).float()
            mu, logvar = model.encode(xb, yb_f)
            zs.append(mu.cpu().numpy())
            ys.append(yb.cpu().numpy())
    return np.concatenate(zs), np.concatenate(ys).ravel()

def visualize_tsne(z, y, label_encoder):
    z_emb = TSNE(n_components=2, random_state=42).fit_transform(z)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1], hue=label_encoder.inverse_transform(y), s=10)
    plt.title("t-SNE of Latent Space")
    plt.show()

def evaluate_classification(z_train, y_train, z_test, y_test, label_encoder):
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    rf.fit(z_train, y_train)
    y_pred = rf.predict(z_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.inverse_transform(sorted(np.unique(y_test)))))

def evaluate_on_raw(X_train, y_train, X_test, y_test, label_encoder):
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest on Raw Features:")
    print(classification_report(y_test, y_pred,
          target_names=label_encoder.inverse_transform(sorted(np.unique(y_test)))))


def plot_loss(train_history):
    plt.figure()
    plt.plot(train_history["train_loss"], label="train")
    plt.plot(train_history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("VAE Loss")
    plt.show()