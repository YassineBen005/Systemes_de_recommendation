#implémentation de la méthode NMF
import numpy as np
import matplotlib.pyplot as plt
import math
#on récupère la matrices des notes
matrice_originale=np.loadtxt("Matrix.txt",dtype=int)
matrice_originale=np.array(matrice_originale)
def recherche_k_optimal(L):
    n,p=L.shape
    #on crée la matrice de poids où Sij=1 ssi Lij n'est 
    #pas 9 sinon il Sij=0
    S=[[0 for i in range(p)] for j in range(n)]
    for i in range(n):
        for j in range(p):
            if L[i][j]!=9:
                S[i][j]=1
    S=np.array(S)
    nombre_de_notes=np.sum(S)
    #on initialise les matrices de factorisation avec 
    #des valeurs aléatoires
    #on veut minimiser l'erreur qui est  le carré 
    #la norme de la différence entre L et WH
    epsilon=0.00000001
    listes_des_erreurs=[]
    #le k optimal est celui qui réalsie le minimum des erreurs
    for k in range(16,36):
        print(k)
        erreur=1000
        W=np.random.rand(n,k)
        H=np.random.rand(k,p)
        erreur_prec = erreur+10
        i=0#compteur de parcours de boucle
        #on va s'arrêter où l'erreur stagne 
        #à 0.001 près 
        while i<2000 and (erreur_prec - erreur)>0.001:
            erreur_prec=erreur
            H*=np.dot(W.T,S*L)/(np.dot(W.T,S*np.dot(W,H))+epsilon)
            W*=np.dot(S*L,H.T)/(np.dot(S*np.dot(W,H),H.T)+epsilon)
            i+=1
            erreur=np.linalg.norm(S*(L-np.dot(W,H)),'fro')
            erreurfinal=erreur
        listes_des_erreurs.append((k,erreurfinal))
    L=sorted(listes_des_erreurs,key=lambda x : x[1])
    print(L[0][0])
    return L[0][0]
#k_optimal=recherche_k_optimal(matrice_originale) on peut faire ainsi mais cela prend du temps 
# je l'ai fait une seule fois et cela suffit 
k_optimal=13
def methode_NMF(L):
    global k_optimal
    n,p=L.shape
    #on crée la matrice de poids où Sij=1 ssi Lij n'est 
    #pas 9 sinon il Sij=0
    S=[[0 for i in range(p)] for j in range(n)]
    for i in range(n):
        for j in range(p):
            if L[i][j]!=9:
                S[i][j]=1
    S=np.array(S)
    nombre_de_notes=np.sum(S)
    #on initialise les matrices de factorisation avec 
    #des valeurs aléatoires
    #on veut minimiser l'erreur qui est  le carré 
    #la norme de la différence entre L et WH
    epsilon=0.00000001
    k=k_optimal
    erreur=1000
    W=np.random.rand(n,k)
    H=np.random.rand(k,p)
    erreur_prec = erreur+10
    i=0#compteur de parcours de boucle
    while i<2000 and (erreur_prec - erreur)>0.001:
        print(i)
        erreur_prec=erreur
        H*=np.dot(W.T,S*L)/(np.dot(W.T,S*np.dot(W,H))+epsilon)
        W*=np.dot(S*L,H.T)/(np.dot(S*np.dot(W,H),H.T)+epsilon)
        i+=1
        erreur=np.linalg.norm(S*(L-np.dot(W,H)),'fro')
        Wfinale=W.copy()
        Hfinale=H.copy()
        erreurfinal=erreur
    eqm=np.sqrt(erreurfinal**2 / nombre_de_notes)
    print("l'erreur quadratique moyenne est:",eqm)
    Mres=np.dot(Wfinale,Hfinale)
    return Mres
def test_matrice_slicee(Matrice):
    L=Matrice[:1000,:1000]
    n,p=L.shape
    Ltest=L.copy()
    masque=np.random.rand(n,p) < 0.8
    Ltest[masque]=9
    Lres=methode_NMF(Ltest)
    print(L)
    print(Lres)
    print(L==Lres)
    print(np.sum(L==Lres)/(n*p))
def sauvegarde_de_matrice_NMF():
    Mres=methode_NMF(matrice_originale)
    print("methode NMF")
    np.savetxt("matrice_predit_NMF.txt",Mres,fmt='%d',delimiter=' ')
def tracage_fscore():
    L=matrice_originale.copy()
    lignes,cols=np.where(L!=9)
    nb_vals_connues=len(lignes)
    porcentage_caché=int(0.1*nb_vals_connues)
    indices = np.random.choice(nb_vals_connues, porcentage_caché, replace=False)
    L[lignes[indices],cols[indices]]=9
    Mres=methode_NMF(L)
    valeurs_connues=matrice_originale[lignes,cols]
    valeurs_predites=Mres[lignes,cols]
    seuils=np.linspace(1,2,100)
    L_precision=[]
    L_rappel=[]
    for s in seuils :
        #si la note prédite est supérieure au seuil on l'a prédit à deux
        VP=np.sum((valeurs_connues==2)&(valeurs_predites>=s))
        FP=np.sum((valeurs_connues!=2)&(valeurs_predites>=s))
        FN=np.sum((valeurs_connues==2)&(valeurs_predites<s))
        precision=VP/(VP+FP)
        rappel=VP/(VP+FN)
        L_precision.append(precision)
        L_rappel.append(rappel)
    plt.figure()
    plt.plot(L_rappel,L_precision)
    plt.show()
#méthode naive où on remplace touutes les cases non-connues par 
#la moyenne de la ligne (ou de la colonne ) où ils se trouvent 
def methode_naive_moyennes(mat):
    #on calcule la moyenne de la ligne i et la moyenne de la colonne j 
    n,p=L.shape
    moyenne_lignes=[]
    for i in range(n):
        i=0
        compt=0
        for j in range(p):
            if L[i][j]!=9:
                compt+=L[i][j]
                i+=1
        moyenne_lignes.append(compt/i)
    moyenne_cols=[]
    for j in range(p):
        i=0
        compt=0
        for i in range(n):
            if L[i][j]!=9:
                compt+=L[i][j]
                i+=1
        moyenne_cols.append(compt/i)
    #remplissage :
    for i in range(n):
        for j in range(p):
            if L[i][j]==9:
                L[i][j]=moyenne_lignes[i]#ou moyenne_cols[j] selon ce qu'on veut 
    return L
#méthode naive où on remplace toutes les cases non-connues par des 1
def methode_naive_1(mat):
    L=mat.copy()
    n,p=L.shape
    for i in range(n):
        for j in range(p):
            if L[i][j]==9:
                L[i][j]=1
    return L
def erreur(matrice_reel,matrice_predit,liste_idx):
    k = 0
    g = 0
    VP = 0
    FN = 0
    e = []
    n = 0
    for elem in liste_idx:
            i,j = elem
            if matrice_predit[i,j]==2:
                n+=1 
            if matrice_predit[i,j] == matrice_reel[i,j] : 
                k += 1
            if (matrice_reel[i,j] == 2 and matrice_predit[i,j] == 0) or (matrice_reel[i,j] == 0 and matrice_predit[i,j] == 2):
                g+=1   
            if (matrice_reel[i,j]== 2 and matrice_predit[i,j]!= 2):
                FN+=1
            if (matrice_reel[i,j]==2 and matrice_predit[i,j] == 2):
                VP+= 1     
            
    perc = (k/len(liste_idx) )* 100
    grve = (g/len(liste_idx))*100
    rappel = VP/(VP+FN)
    pres = VP/n
    if (rappel+pres) != 0:
        f1score = 2*rappel*pres/(rappel+pres)
    #pour methode_naive_1 on rentre toujous dans cette section else car VP=0
    else :
        f1score=0
    dic={"Taux de reussite :":perc ,"Taux d'erreurs graves :":grve,"reppel :":rappel,"précision :":pres,"f-score :":f1score}
    return dic
#on cache 10% des valeurs connues pour tester les différentes méthodes 
L=matrice_originale.copy()
observed = np.argwhere(L != 9 )
n_observed = len(observed)
n_to_hide = int(0.1 * n_observed)
perm = np.random.permutation(n_observed)
hide = observed[perm[:n_to_hide]]
R = L.copy()
for (i,j) in hide:
    R[i,j] = 9 
sauvegarde_de_matrice_NMF()






