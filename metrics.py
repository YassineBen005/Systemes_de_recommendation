import numpy as np


def metrics(mat_reel, mat_predite, liste_idx):
    Dico = {"Proportion correcte":0, "Erreurs graves":0, "Précision(classe2)" : 0, "Recall(classe2)":0, "F-1score(classe2)":0}
    VP,FP,FN = 0,0,0
    for i,j in liste_idx:
        if mat_reel[i,j] == mat_predite[i,j]:
            Dico["Proportion correcte"]+=1
        if mat_reel[i,j] == 2 and mat_predite[i,j]!= 2:
            FN+=1
        if mat_reel[i,j] ==2 and mat_predite[i,j] == 2:
            VP+=1
        if mat_reel[i,j] !=2 and mat_predite[i,j] == 2:
            FP+=1
        if (mat_reel[i,j] ==0 and mat_predite[i,j] == 2) or (mat_reel[i,j] ==2 and mat_predite[i,j] == 0):
            Dico["Erreurs graves"]+=1    
    Dico["Proportion correcte"]/= len(liste_idx)                   
    Dico["Erreurs graves"]/= len(liste_idx)
    if VP != 0 :
        Dico["Précision(classe2)"] = VP/(VP+FP)
        Dico["Recall(classe2)"] = VP/(VP+FN)
        Dico["F-1score(classe2)"] = 2*(Dico["Recall(classe2)"]*Dico["Précision(classe2)"])/(Dico["Recall(classe2)"] + Dico["Précision(classe2)"])
    return Dico

def masque(mat, pourcentage):
    if pourcentage>1:
        pourcentage/=100
    observed = np.argwhere(mat != 9 )
    n_observed = len(observed)
    n_to_hide = int(pourcentage* n_observed)
    perm = np.random.permutation(n_observed)
    hide = observed[perm[:n_to_hide]]
    R = mat.copy()
    for (i,j) in hide :
        R[i,j] = 9 

    return R,hide    
def round_seuil (nombre,seuil):
    if nombre >= seuil : 
        return 2
    if nombre < 0.5 : 
        return 0
    return 1 

