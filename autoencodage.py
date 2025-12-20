import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



# On charge la matrice depuis Matrix.txt
R = np.loadtxt("Matrix.txt", dtype=np.float32)  # (n_users, n_items)

print("Shape de R :", R.shape)

# On masque des positions connues (notes observées) 
#M_i,j=1 si l'utilisateur i a noté le livre j et 0 sinon
M = (R != 9).astype(np.float32)

# On remplace les 9 par 0 pour ne pas les pénaliser dans la perte
R[R == 9] = 0.0

print("Nb de notes connues :", M.sum())

'''
On va utiliser dans ce projet des tenseurs Pytorch pour qu'on puisse calculer automatiquement les gradients utils,
et pour optimiser les calcules vectoriels.
'''


R_tensor = torch.from_numpy(R)        # shape (n_users, n_items)
M_tensor = torch.from_numpy(M)        # même shape
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") (Au cas où on trouve un PC avec un GPU CUDA)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #Car je suis sur un Mac Apple Silicon M2

R_tensor = R_tensor.to(device)
M_tensor = M_tensor.to(device)

class RatingsDataset(Dataset):
    def __init__(self, R, M):
        self.R = R  # (n_users, n_items)
        self.M = M

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx):
        return self.R[idx], self.M[idx]
    
'''
La classe RatingsDataset encapsule la matrice de notes et son masque.
Chaque élément du Dataset correspond à un utilisateur : un vecteur de notes R[u] et le masque M[u].
PyTorch utilise __getitem__ pour extraire automatiquement les entrées nécessaires à l'entraînement et __len__ pour connaître la taille du dataset.
Cela permet d utiliser un DataLoader pour gérer les batchs, le mélange aléatoire et l optimisation du modèle.
'''


dataset = RatingsDataset(R_tensor, M_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self, n_items, latent_dim=64):
        super().__init__() #Permet à Pytorch de bien initialiser le module
        
        self.encoder = nn.Sequential(
            nn.Linear(n_items, 512), #On prend un vecteur de taille 2243, et on le transforme en un vecteur de taille 512
            #On prend en entrée un vecteur de taille 2243 x puis on applique la transformation affine a=Wx+b avec W la matrice des poids et b le vecteur du biais
            #On réduit la dimension de x mais pas trop vite pour capturer le plus d'informations que possible
            
            nn.ReLU(),#Fonction d'activation ReLu(x)=max(x,0)
            nn.Linear(512, latent_dim),#On réduit la dimension encore à 64 chiffres, c'est la représentation latente de l'utilisateur
            #La représentation latente d'un utilisateur regroupe ces préférences en 64 catégories au lieu des 2243 initiales
            nn.ReLU()
            
        )
        
        self.decoder = nn.Sequential( #le décodeur fait l'opération inverse de l'encodeur
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_items)
            # pas d'activation finale car on veut prédire des valuers continues qui seront ensuite arrondies à (0,1,2)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    



#On construit l'autoencodeur
n_items = R_tensor.shape[1]
model = AutoEncoder(n_items=n_items, latent_dim=64)
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#La boucle d'entraînement
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    
    for batch_R, batch_M in dataloader:
        batch_R = batch_R.to(device)
        batch_M = batch_M.to(device)
        pred = model(batch_R)
        loss = masked_mse(pred, batch_R, batch_M)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss : {total_loss:.4f}")


#On sauvegarde les poids appris lors de l'entraînement pour les réutiliser plus tard
#state_dict() contient l’ensemble des paramètres optimisés (matrices de poids et vecteurs de biais) de toutes les couches du modèle.
torch.save(model.state_dict(), "autoencoder_trained.pth")
print("Modèle sauvegardé dans autoencoder_trained.pth")

