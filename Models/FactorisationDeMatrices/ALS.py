#Code python pour l'implémentation de la méthode de factorisation ALS
import numpy as np
import matplotlib.pyplot as plt
import metrics as met 
max_iterations = 20
lambda_reg = 0.1
#Calcul de la fonction de coût qu'on veut minmiser
def cost(U, V, R, mask, lambda_reg):
    prediction = np.dot(U, V.T)
    # Erreur sur les éléments observés uniquement
    base_cost = np.sum((R[mask] - prediction[mask])**2)
    # Régularisation (norme de Frobenius)
    reg_cost = lambda_reg * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)
    return base_cost + reg_cost
#Les fonctions qui vont modifier U et V 
def update_U(U, V, R, mask, lambda_reg,k):
    m,n = R.shape
    for i in range(m):
        # Indices des items notés par l'utilisateur i
        idx = mask[i] 
        if np.sum(idx) > 0: # Si l'utilisateur a noté au moins un truc
            V_s = V[idx]   # Sous-ensemble de V correspondant aux items notés
            R_s = R[i, idx] # Sous-ensemble des notes
            
            # A = V_s^T * V_s + lambda * I
            A = np.dot(V_s.T, V_s) + lambda_reg * np.eye(k)
            # B = V_s^T * R_s
            B = np.dot(V_s.T, R_s)
            
            U[i] = np.linalg.solve(A, B)
    return U
def update_V(U, V, R, mask, lambda_reg,k):
    m, n = R.shape
    for j in range(n):
        # Indices des utilisateurs ayant noté l'item j
        idx = mask[:, j]
        if np.sum(idx) > 0:
            U_s = U[idx]
            R_s = R[idx, j]
            
            A = np.dot(U_s.T, U_s) + lambda_reg * np.eye(k)
            B = np.dot(U_s.T, R_s)
            
            V[j] = np.linalg.solve(A, B)
    return V


def ALS(R, epsilon):
    global IDX , listes_perf,Listes_indices
    m,n = R.shape
    for k in [8]:
        U,V = np.random.randn(m, k)*0.1, np.random.randn(n, k)*0.1
        previous_cost = cost(U,V,R, IDXREMPLIE, lambda_reg)
        for iter in range(max_iterations):
            print(f"Actuellement a l'iteration :{iter}")
            U = update_U(U,V, R, IDXREMPLIE, lambda_reg,k)
            V = update_V(U,V, R, IDXREMPLIE, lambda_reg,k)
            current_cost = cost(U,V,R, IDXREMPLIE, lambda_reg)
            if abs(previous_cost - current_cost) < epsilon :
                print(f"Convergence à l'itération {iter}")
                break
            previous_cost = current_cost
        e = np.dot(U, V.T)
    return e     



L = np.loadtxt("Matrix.txt", dtype=int)

R,hide = met.masque(L,0.1)
IDXREMPLIE = (L != 9)

n,m = L.shape
#Résultats de la simulation
#
#P = ALS(R,0.001)
#for i in range(n) :
#    for j in range(m): 
#        if P[i,j ] > 2 :
#            P[i,j ]  = 2 
#        P[i,j] = met.round_seuil(P[i,j],1.7)
#print(met.metrics(L,P,hide))    
#Application de l algorithme sur la matrice :
P = ALS(L,0.0001)  
for i in range(n) :
    for j in range(m): 
        if P[i,j ] > 2 :
            P[i,j ]  = 2 
        if IDXREMPLIE[i,j]:#Pour un meilleur taux de proportion correcte
#il faut revérifier les valeurs dèja connues et les corriger s'il y'a eu des erreurs
            P[i,j] = L[i,j]    
        P[i,j] = met.round_seuil(P[i,j],1.7)  
            




#print(P[0])    
np.savetxt("Matrice_predit_ALS.txt",P,fmt='%d', delimiter=' ')








