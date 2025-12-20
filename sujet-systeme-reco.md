# Projet – Factorisation de matrices pour les systèmes de recommandation

**Contact: Olivier François (olivier.francois@univ-grenoble-alpes.fr)**

## Contexte

La bibliothèque de la petite ville de **Claireflasque**, située dans les **Cévennes jurassiennes**, possède **2243 livres**.
Un livre est une technologie sans fil permettant de transmettre des histoires ou des connaissances au cerveau, sans émission de CO2.

La bibliothèque compte **1403 abonnés réguliers**, lecteurs passionnés et assidus depuis plusieurs années.

Après de longues études en informatique théorique, **Alan** et **Kurt** sont devenus bibliothécaires à Claireflasque.
Dans un fichier de retour des ouvrages, ils ont consciencieusement noté les avis des abonnés sur chaque livre consulté :

* `0` : « Bof, je n’ai pas trop apprécié ce livre »
* `1` : « Ce livre fait bien passer le temps »
* `2` : « C’était un très bon livre »

Les distractions étant rares, et la pluie fréquente dans les Cévennes jurassiennes, chaque abonné n’a pu lire et noter qu’un nombre limité d’ouvrages — environ **un quart** de la collection, en moyenne.
Dans le fichier, les avis manquants sont codés par la valeur **9**.



## Problématique

S’inspirant d’un concours organisé dans la ville voisine de **Netteflisque**, Alan et Kurt proposent un **prix** (l’ouvrage *Astérix à l’Imag*) à quiconque saura les aider à fournir des recommandations de lecture pertinentes à leurs abonnés.

Sur le principe du fameux « *On pense que vous pourriez aimer...* », les recommandations devront s’appuyer sur les avis déjà enregistrés dans le fichier de retours.

L’objectif du challenge est donc de **compléter le fichier** en attribuant une valeur `0`, `1` ou `2` à chaque ouvrage non lu (valeurs initialement `9`).
Ces valeurs constitueront des **recommandations personnalisées** pour chaque abonné.



## Objectifs et évaluation du projet

Pour mener à bien ce projet, un groupe devra 

1. **Se documenter** sur les méthodes utilisées dans les systèmes de recommandation, en particulier sur les **algorithmes basés sur la factorisation matricielle**.

2. **Comparer** plusieurs approches, y compris des **méthodes naïves** (remplacement par la moyenne, la médiane, etc.) 
et d’autres méthodes issues de l’état de l’art. Chaque membre du groupe explorera une méthode différente de celles
choisies par les autres membres du groupe.  

3. **Implémenter** plusieurs **méthodes de factorisation matricielle** en :

   * expliquant leurs principes algorithmiques,
   * présentant la démarche de sélection des méthodes (hypothèses, avantages, inconvénients).

4. **Évaluer les résultats** en masquant une partie des avis connus, puis en comparant les prédictions aux vraies valeurs.
   Deux scores sont à calculer :

   * le **pourcentage d’erreurs de prédiction**,
   * le **pourcentage d’erreurs “graves”**, correspondant aux cas où l’on prédit qu’un abonné aimera beaucoup un livre alors qu’il ne l’aime pas (ou inversement).

5. Proposer finalement des **recommandations pour tous les abonnés**, c’est-à-dire une **matrice complète** où toutes les valeurs `9` auront été remplacées par `0`, `1` ou `2`.


Le projet étant **non supervisé**, chaque groupe est libre de choisir les méthodes à implémenter, à condition de **justifier clairement ses choix**.
Chacun pourra **utiliser son langage de programmation favori** (langages recommandés : **Python** ou **R**). L’implémentation peut s’appuyer sur des bibliothèques existantes.
Le résultat final sera **évalué par un oracle** connaissant l’avis futur de chaque abonné grâce à un **sacrifice rituel de poulet à la déesse Alphabetis**.
Cet oracle pourra être consulté **deux fois** au cours du projet (prévoir les poulets).





## Données

Le fichier de retours incomplet est une **matrice de taille 1403 × 2243**, comportant des notes (`0`, `1`, `2`) et environ **2,4 millions de valeurs manquantes** (`9`).
Ce fichier sera mis à disposition au lancement du projet.



## Références bibliographiques

Constituer une **brève bibliographie** (5 à 10 références) fait partie intégrante du projet.

On pourra naturellement commencer par interroger une IA générative et consulter Wikipédia :
[Factorisation de matrices pour les systèmes de recommandation](https://fr.wikipedia.org/wiki/Factorisation_de_matrices_pour_les_syst%C3%A8mes_de_recommandation)


