# Projet Math appliqué - Équipe 26

Ce dépôt contient l'implémentation de différentes approches de filtrage collaboratif pour prédire les notes d'utilisateurs sur une base de données de livres parsemée (75 % de données manquantes).

##  Structure du Projet

Le projet est organisé autour de plusieurs méthodes d'approximation et d'optimisation pour la complétion de matrice :

### 1. Méthodes Naïves (Ligne de base)
* `methode_par_remplissage_par_1.ipynb` : Remplissage constant avec la valeur 1.
* `Methode_Naive par moyennes colonnes:lignes.ipynb` : Remplissage basé sur les moyennes des utilisateurs et des items.

### 2. Factorisation de Matrice (Matrix Factorization)
* `ALS.py` : Optimisation par moindres carrés alternés (Alternating Least Squares). 
* `SGD.ipynb` : Optimisation par descente de gradient stochastique par mini-lots (Mini-Batch SGD) avec biais.
* `NMF.py` : Factorisation de matrice non-négative.

### 3. Apprentissage Profond (Deep Learning)
* `autoencodage.py` : Methode d'autoencodage avec parametres fixes .
* `autoencodage_version_2.py` : Methode d'autoencodage avec seuillage adaptif , avec implementation de la distinction entres les vrai 0 et les valeurs manquantes.
* `autoencodage_version_finale.py` : Methode d'autoencodage avec Seuillage adaptif , avec poids variables des differentes classes selon leurs fréquences.

### 4. Évaluation et Utilitaires
* `metrics.py` : Script de calcul des métriques de performance (Accuracy, Précision, Rappel, F1-Score).
* `Matrix.txt` : Jeu de données original (données manquantes marquées par le chiffre 9).
* `Ref.bib` : Références bibliographiques du projet.
* `Rapport.tex` : Rapport du projet redigé en Latex.

## Configuration et Utilisation

### Prérequis
* Python 3.x
* Bibliothèques : `numpy`, `matplotlib`, `scikit-learn` ( Optionnel pour verification de la courbe PR dans la methode SGD )

### Exécution
1. Assurez-vous que le fichier `Matrix.txt` est présent dans le répertoire racine.
2. Executez les fichier concerné pour generer la matrice remplis avec la methode choisis.