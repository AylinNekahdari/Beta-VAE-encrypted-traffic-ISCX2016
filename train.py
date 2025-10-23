import torch
from model import BetaCVAE, beta_vae_loss
from config import *
from utils import load_and_preprocess
from dataset import make_dataloaders

def train():
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, le = load_and_preprocess(DATA_PATH)
    train_loader, val_loader, test_loader = make_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE)

    model = BetaCVAE(input_channels=1, seq_len=X_train.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_history = {"train_loss": [], "val_loss": []}
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1).float()
            opt.zero_grad()
            x_recon, mu, logvar = model(xb, yb)
            loss, _, _ = beta_vae_loss(xb, x_recon, mu, logvar, beta=BETA)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_history["train_loss"].append(train_loss)


    # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb_f = yb.to(DEVICE).unsqueeze(1).float()
                x_recon, mu, logvar = model(xb, yb_f)
                loss, _, _ = beta_vae_loss(xb, x_recon, mu, logvar, beta=BETA)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        train_history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")


    torch.save(model.state_dict(), f"{MODEL_DIR}/beta_cvae.pth")
    print("✅ Model saved.")
    return model, train_history, (X_test, y_test, scaler, le, test_loader)
