import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



# On charge la matrice depuis Matrix.txt
R = np.loadtxt("Matrix.txt", dtype=np.float32)  # (n_users, n_items)

print("Shape de R :", R.shape)

# On garde une copie complète pour l'évaluation (toutes les vraies notes)
R_full = R.copy()

# On masque des positions connues (notes observées) 
#M_i,j=1 si l'utilisateur i a noté le livre j et 0 sinon
M_full = (R != 9).astype(np.float32)

print("Nb de notes connues (total) :", M_full.sum())

# On choisit aléatoirement une partie des notes connues à cacher (10 %)
np.random.seed(0)
known_indices = np.argwhere(M_full == 1)     # toutes les paires (u,i) avec note connue
n_known = known_indices.shape[0]
test_ratio = 0.1
n_test = int(test_ratio * n_known)

perm = np.random.permutation(n_known)
test_idx = known_indices[perm[:n_test]]      # indices (u,i) du test
train_idx = known_indices[perm[n_test:]]     # indices utilisés pour l'entraînement

# On crée une matrice d'entraînement où les notes de test sont cachées (remplacées par 9)
R_train = R_full.copy()
for u, i in test_idx:
    R_train[u, i] = 9.0

# Nouveau masque d'entraînement : 1 si note visible pendant l'entraînement
M = (R_train != 9).astype(np.float32)

# On crée la matrice cible pour l'entraînement : 0/1/2, 0 pour les manquants (dont les notes de test)
R_target = R_train.copy()
R_target[R_target == 9] = 0.0

print("Nb de notes connues (entraînement) :", M.sum())

# À partir de maintenant, on travaille avec R_train / R_target / M pour l'autoencodeur
R = R_train


# V2 : encodage "one-hot" pour distinguer vrais 0 des notes manquantes
# Pour chaque case R[i,j], on crée un vecteur de taille 4 :
# [1,0,0,0] si R[i,j] = 0
# [0,1,0,0] si R[i,j] = 1
# [0,0,1,0] si R[i,j] = 2
# [0,0,0,1] si R[i,j] = 9 (note manquante)
# On obtient une entrée de dimension 4 * n_items au lieu de n_items.
#C'est la différence fondamentale avec la version 1

n_users, n_items = R.shape

is0     = (R == 0).astype(np.float32)
is1     = (R == 1).astype(np.float32)
is2     = (R == 2).astype(np.float32)
ismiss  = (R == 9).astype(np.float32)

# shape (n_users, n_items, 4)
R_onehot = np.stack([is0, is1, is2, ismiss], axis=-1)

# On aplati les 4 canaux par item en un grand vecteur de taille 4 * n_items
R_input = R_onehot.reshape(n_users, 4 * n_items).astype(np.float32)

'''
On va utiliser dans ce projet des tenseurs Pytorch pour qu'on puisse calculer automatiquement les gradients utils,
et pour optimiser les calcules vectoriels.
'''

R_input_tensor  = torch.from_numpy(R_input)    # (n_users, 4 * n_items)
R_target_tensor = torch.from_numpy(R_target)   # (n_users, n_items)
M_tensor        = torch.from_numpy(M)          # même shape que R_target

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #(Au cas où on trouve un PC avec un GPU CUDA)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #Car je suis sur un Mac Apple Silicon M2

R_input_tensor  = R_input_tensor.to(device)
R_target_tensor = R_target_tensor.to(device)
M_tensor        = M_tensor.to(device)

class RatingsDataset(Dataset):
    def __init__(self, R_in, R_true, M):
        self.R_in   = R_in    # (n_users, 4 * n_items)  - entrée du modèle
        self.R_true = R_true  # (n_users, n_items)      - cible (0/1/2, 0 pour les manquants)
        self.M      = M       # (n_users, n_items)      - masque des notes connues

    def __len__(self):
        return self.R_in.shape[0]

    def __getitem__(self, idx):
        return self.R_in[idx], self.R_true[idx], self.M[idx]
    
'''
La classe RatingsDataset encapsule la matrice de notes et son masque.
Chaque élément du Dataset correspond à un utilisateur : un vecteur de notes R[u] et le masque M[u].
PyTorch utilise __getitem__ pour extraire automatiquement les entrées nécessaires à l'entraînement et __len__ pour connaître la taille du dataset.
Cela permet d utiliser un DataLoader pour gérer les batchs, le mélange aléatoire et l optimisation du modèle.
'''

dataset = RatingsDataset(R_input_tensor, R_target_tensor, M_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self, n_in, n_out, latent_dim=64):
        super().__init__() #Permet à Pytorch de bien initialiser le module
        
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 512), #On prend un vecteur de taille 2243, et on le transforme en un vecteur de taille 512
            #On prend en entrée un vecteur de taille 2243 x puis on applique la transformation affine a=Wx+b avec W la matrice des poids et b le vecteur du biais
            #On réduit la dimension de x mais pas trop vite pour capturer le plus d'informations que possible
            nn.Dropout(0.15), #on oublie 15% des neurones pour éviter le surapprentissage et la mémorisation
            nn.ReLU(),#Fonction d'activation ReLu(x)=max(x,0)
            nn.Linear(512, latent_dim),#On réduit la dimension encore à 64 chiffres, c'est la représentation latente de l'utilisateur
            #La représentation latente d'un utilisateur regroupe ces préférences en 64 catégories au lieu des 2243 initiales
            nn.ReLU()
            
        )
        
        self.decoder = nn.Sequential( #le décodeur fait l'opération inverse de l'encodeur
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, n_out)
            # pas d'activation 
            #finale car on veut prédire des valuers continues qui seront ensuite arrondies à (0,1,2)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    


#On construit l'autoencodeur
#En V2, la dimension d'entrée est 4 * n_items, mais la sortie reste de taille n_items
n_items_in  = R_input_tensor.shape[1]    # 4 * n_items réels
n_items_out = R_target_tensor.shape[1]   # n_items réels (2243)

model = AutoEncoder(n_in=n_items_in, n_out=n_items_out, latent_dim=64)
model = model.to(device)


#On définit la fonction de perte masquée
'''
On calcule la moyenne des erreurs quadratiques uniquement sur les notes observées,
en ignorant complètement les prédictions faites pour les livres non notés.
'''
def masked_mse(predicted_value, true_value, mask):
    diff = (predicted_value - true_value) * mask
    mse = (diff ** 2).sum() / mask.sum()
    return mse


#On définit l'optimizeur ADAM: Adaptive Moment Estimation, c'est une descente du gradient adaptive. Il assure une convergence rapide et sûre.
#Il va minimizer la fonction de perte

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#La boucle d'entraînement
epochs = 150

for epoch in range(epochs):
    total_loss = 0
    
    for batch_in, batch_true, batch_M in dataloader:
        batch_in   = batch_in.to(device)
        batch_true = batch_true.to(device)
        batch_M    = batch_M.to(device)

        pred = model(batch_in)               # shape (batch_size, n_items)
        loss = masked_mse(pred, batch_true, batch_M)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss : {total_loss:.4f}")


#On sauvegarde les poids appris lors de l'entraînement pour les réutiliser plus tard
#state_dict() contient l’ensemble des paramètres optimisés (matrices de poids et vecteurs de biais) de toutes les couches du modèle.
torch.save(model.state_dict(), "autoencoder_trained_v2.pth")
print("Modèle V2 sauvegardé dans autoencoder_trained_v2.pth")




# ÉVALUATION DE LA VERSION 2
model.eval()
with torch.no_grad():
    # Prédiction pour tous les utilisateurs à partir de R_input_tensor
    R_pred_tensor = model(R_input_tensor)          # shape (n_users, n_items)
    R_pred = R_pred_tensor.cpu().numpy()

# On borne les prédictions dans [0,2] puis on arrondit à {0,1,2}
R_pred_clipped = np.clip(R_pred, 0, 2)
R_pred_rounded = np.rint(R_pred_clipped).astype(int)

# On évalue UNIQUEMENT sur les notes qui avaient été cachées (test_idx)
y_true = []
y_pred = []

for (u, i) in test_idx:
    true_rating = int(R_full[u, i])            # vraie note (0/1/2)
    pred_rating = int(R_pred_rounded[u, i])    # note prédite arrondie
    y_true.append(true_rating)
    y_pred.append(pred_rating)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Accuracy
accuracy = (y_true == y_pred).mean()
error_rate = 1.0 - accuracy

# Erreurs graves : 0 prédit 2 ou vice versa
severe_mask = ((y_true == 0) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 0))
severe_error_rate = severe_mask.mean()

# F1-score pour la classe 2 (bons livres)
tp = np.sum((y_true == 2) & (y_pred == 2))
fp = np.sum((y_true != 2) & (y_pred == 2))
fn = np.sum((y_true == 2) & (y_pred != 2))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("\nTEST DE LA V2")
print(f"Accuracy sur les notes cachées       : {accuracy*100:.2f} %")
print(f"Erreurs graves (0↔2) sur test        : {severe_error_rate*100:.2f} %")
print(f"F1-score pour la classe 2 (test)     : {f1_score:.4f}")



