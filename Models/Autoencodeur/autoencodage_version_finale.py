import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import time

#Device à utiliser
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device : Apple MPS (Metal)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device : GPU CUDA")
else:
    device = torch.device("cpu")
    print("Device : CPU")

def run_one_test_ratio(test_ratio):
    
    print(f"\n" + "="*50)
    print(f"DÉMARRAGE DU TEST : Ratio {test_ratio*100:.0f}% ")
    print("="*50)
    
    print("[1/6] Chargement des données...")
    
    R = np.loadtxt("Matrix.txt", dtype=np.float32)
    R_full = R.copy()
    M_full = (R != 9).astype(np.float32)

    np.random.seed(42)
    known_indices = np.argwhere(M_full == 1)
    n_known = known_indices.shape[0]
    n_test = int(test_ratio * n_known)

    perm = np.random.permutation(n_known)
    test_idx = known_indices[perm[:n_test]]      # Indices test

    # Création de la matrice d'entraînement masquée
    R_train = R_full.copy()
    for u, i in test_idx:
        R_train[u, i] = 9.0

    M = (R_train != 9).astype(np.float32)
    
    # Cible : On remplace les 9 par 0, mais le masque M gérera l'apprentissage
    R_target = R_train.copy()
    R_target[R_target == 9] = 0.0

    # Poids pour équilibrer la Loss (MSE pondérée)
    train_vals = R_target[M == 1].astype(int)
    counts = np.bincount(train_vals, minlength=3).astype(np.float32)
    class_weights = counts.sum() / (3.0 * (counts + 1e-8))
    
    # On crée une matrice de poids W de la même taille que R
    # Chaque case (u,i) aura le poids correspondant à sa classe (0, 1 ou 2)
    W = np.ones_like(R_target)
    W[R_target == 0] = class_weights[0]
    W[R_target == 1] = class_weights[1]
    W[R_target == 2] = class_weights[2]
    # Les données manquantes n'ont pas d'importance (le masque M les annulera), 
    # mais on laisse un poids par défaut.

    print(f"      -> {n_known} notes connues.")
    print(f"      -> Poids calculés : {class_weights}")

    # Encodage ONE-HOT (4 canaux)
    print("[2/6] Encodage One-Hot...")
    n_users, n_items = R_train.shape
    is0     = (R_train == 0).astype(np.float32)
    is1     = (R_train == 1).astype(np.float32)
    is2     = (R_train == 2).astype(np.float32)
    ismiss  = (R_train == 9).astype(np.float32)

    R_onehot = np.stack([is0, is1, is2, ismiss], axis=-1)
    R_input = R_onehot.reshape(n_users, 4 * n_items).astype(np.float32)

    # Conversion Tenseurs
    R_input_tensor  = torch.from_numpy(R_input).to(device)
    R_target_tensor = torch.from_numpy(R_target).to(device)
    M_tensor        = torch.from_numpy(M).to(device)
    W_tensor        = torch.from_numpy(W).float().to(device)

    class RatingsDataset(Dataset):
        def __init__(self, R_in, R_true, M, W):
            self.R_in = R_in    
            self.R_true = R_true  
            self.M = M 
            self.W = W      
        def __len__(self):
            return self.R_in.shape[0]
        def __getitem__(self, idx):
            return self.R_in[idx], self.R_true[idx], self.M[idx], self.W[idx]

    dataset = RatingsDataset(R_input_tensor, R_target_tensor, M_tensor, W_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("[3/6] Construction du modèle...")
    
    class AutoEncoderRegression(nn.Module):
        def __init__(self, n_in, n_out, latent_dim=128):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(n_in, 512),
                nn.Dropout(0.67),#eviter la mémorisation
                nn.ReLU(),
                nn.Linear(512, latent_dim),
                nn.Tanh()
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.67),
                nn.Linear(512, n_out),
            )
        
        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out

    n_items_in  = R_input_tensor.shape[1]
    n_items_out = R_target_tensor.shape[1]

    model = AutoEncoderRegression(n_in=n_items_in, n_out=n_items_out, latent_dim=128)
    model = model.to(device)

    # Fonction de perte MSE Masquée et Pondérée
    def masked_weighted_mse(pred, target, mask, weights):
        diff = (pred - target) * mask
        mse = ((diff ** 2) * weights).sum() / ((mask * weights).sum() + 1e-8)
        return mse

    print("[4/6] Entraînement en cours...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-5)
    epochs = 350 

    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0
        
        for batch_in, batch_true, batch_M, batch_W in dataloader:
            
            pred = model(batch_in)
            loss = masked_weighted_mse(pred, batch_true, batch_M, batch_W)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"      -> Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Temps: {elapsed:.1f}s")

    print(f"      -> Fin de l'entraînement.")

    # --- 6. OPTIMISATION DES SEUILS (SUR LE TRAIN) ---
    print("[5/6] Optimisation des seuils (t0, t1)...")
    
    model.eval()
    with torch.no_grad():
        # On prédit sur tout le dataset
        full_pred = model(R_input_tensor).cpu().numpy()
        full_pred = np.clip(full_pred, 0, 2)
        full_target = R_target_tensor.cpu().numpy()
        full_mask = M_tensor.cpu().numpy()

    # On ne regarde que ce qui était connu dans le TRAIN
    train_true = full_target[full_mask == 1]
    train_pred = full_pred[full_mask == 1]

    # Recherche exhaustive des meilleurs t0 et t1
    candidates = np.linspace(0.0, 2.0, 101) # On teste 100 coupures possibles
    best_acc = -1.0
    best_t0, best_t1 = 0.5, 1.5

    for t0 in candidates:
        for t1 in candidates:
            if t1 <= t0 + 0.1: # On impose un petit écart min
                continue
            
            # On simule la classification
            temp_cls = np.zeros_like(train_pred, dtype=int)
            temp_cls[(train_pred >= t0) & (train_pred < t1)] = 1
            temp_cls[train_pred >= t1] = 2
            
            acc = (temp_cls == train_true).mean()
            if acc > best_acc:
                best_acc = acc
                best_t0 = t0
                best_t1 = t1

    print(f"      -> Meilleurs Seuils trouvés : t0={best_t0:.2f}, t1={best_t1:.2f} (Acc Train: {best_acc*100:.2f}%)")

    print("[6/6] Évaluation finale sur les données cachées...")

    # On applique les seuils trouvés sur les prédictions
    R_pred_final = np.zeros_like(full_pred, dtype=int)
    R_pred_final[(full_pred >= best_t0) & (full_pred < best_t1)] = 1
    R_pred_final[full_pred >= best_t1] = 2

    # --- COMPLETION DE LA MATRICE : on remplace uniquement les 9 ---
    R_completed = R_full.copy().astype(int)
    missing_mask = (R_full == 9)
    R_completed[missing_mask] = R_pred_final[missing_mask]

    np.savetxt("Matrix_completed_autoencodeur.txt", R_completed, fmt="%d")
    print("Matrice complétée sauvegardée dans Matrix_completed.txt")

    # On retourne quand même quelque chose pour ne pas casser l'appel
    return 0.0, 0.0, 0.0


# --- BOUCLE PRINCIPALE ---
#percents = [1, 5, 10, 20] #J'ai testé sur plusieurs pourcentage pour savoir s'il avait un impact, conclusion: pas d'impact
percents = [0]

ratios = [p / 100 for p in percents]

acc_list = []
sev_list = []
f1_list = []

print("\n=== Lancement du Benchmark ===")
start_global = time.time()

for prc, r in zip(percents, ratios):
    acc, sev, f1 = run_one_test_ratio(r)
    acc_list.append(acc)
    sev_list.append(sev)
    f1_list.append(f1)
    print(f"RÉSULTAT ({prc}%) -> Accuracy: {acc*100:.2f}% | Erreurs Graves: {sev*100:.2f}% | F1: {f1:.4f}")

print(f"\nTemps total d'exécution : {time.time() - start_global:.1f}s")
