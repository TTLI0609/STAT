import Projet_Bioinfo
import math
import matplotlib.pyplot as plt
import numpy as np

#==========================
#PARTIE 2 :Préliminaires : données et lecture des fichiers
#=========================

#----------Question 3--------------
def logproba(liste_entiers, m):
    """
    :param liste_entiers: liste d'entiers correpondant à une suite de lettres
    :param m: liste de 4 éléments correspondant au fréquence des lettres A, C, G et T
    :return: retourne la log-probabilité d’une séquence liste_entiers étant donné les fréquences des lettres m
    """
    res = 0
    for i in liste_entiers:
        res += math.log(m[i])
    return res


#----------Question 4--------------
def logprobafast(list_nucleo_count, m):
    """
    on fait la même chose que logproba mais au lieu de prendre toute la séquence comme une liste, 
    on prend seulement en paramètre le résultat de nucleotide_count
    :param list_nucleo_count: dictionnaire associant une lettre à son nombre d'occurence dans une séquence de lettre
    """
    res = 0
    for i in range(len(list_nucleo_count)):
        res += list_nucleo_count[i] * math.log(m[i])
    return res


#==========================
#PARTIE 3 : Annotation des régions promoteurs
#=========================
    
#-------------------------
#       3.1 Description Empirique, préliminaires
#--------------------------

#----------Question 1-------------
  
def code(m, k):
    """
    :param m: un mot préalablement converti en entier 
    :param k: longueur du mot m
    :return: l'indice du mot m dans le tableau ordonné lexicographiquement
    """
    
    res = 0
    for chiffre in m:
        k -= 1
        res += chiffre * (4 **(k))
    return res

    
def decode(i,k):
    """
    :param i: l'indice de la séquence qu'on cherche à décoder dans le tableau ordonné lexicographiquement
    :param k: longueur de la séquence qu'on cherche à décoder
    :return: renvoie la séquence de longueur k correspondante ( sous forme d'une liste d'entierx )
    """
    res = []
    for j in range (k, 0,-1):
        q = i // (4**(j-1))
        r = i % (4**(j-1))
        res.append(q)
        i=r
    return res

 
def comptage_observe(k, seq, possibles):
    """
    :param k: taille des occurences qu'on cherche
    :param seq: une séquence d'ADN en tableau de chiffre
    :param possibles: tableau de mots possibles de longueru k
     compte le nombre d’occurrences pour tous les mots de taille k dans une séquence d’ADN
    """
    dic = dict((key, 0) for key in possibles)    #initialisation du dict pour tout les mots possibles
    for i in range(0, len(seq)-k+1):
        val = code(seq[i:i+k],k)
        if val >=0:
            dic[val]+=1
    return dic


#----------Question 2-------------
            
def mots_possibles(k):
    """
    :param k: longuer des mots qu'on veut générer
    :return: un tableau correspondant aux entiers possibles (en décimal)
    """
    return [x for x in range(4**k)]

#print(mots_possibles(2))
    

def comptage_attendu(k, possibles, freq, nb):
    """
    :param k: taille des occurences qu'on cherche
    :param possibles: tableau de mots possibles de longueru k
    :freq : tuple de 4 éléments contenant la frequence attendue de chaque lettre
    :nb : nombre total de nucléotides dans la séquence
    :return: un dictionnaire associant les mots possibles et leur comptage attendu
    """
    dico = dict((key, 0) for key in possibles)
    for mot in possibles:   
     #   seq = decode(mot,k)
      #  dico[mot] = (math.exp(logproba(seq, freq)) * (1 - (k-1)))
      proba = 1       
      mot_tab = decode(mot,k)
      for chiffre in mot_tab:
          proba *= freq[chiffre]
      dico[mot] = proba*int(nb)
    
    return dico



#----------Question 3-------------
    
def graphe_attendu_observe(nom_seq, sequence, k):
    """
    Affiche un graphe avec le nombre attendu d’occurrences sur l’axe des abcisses 
    et le nombre observé sur l’axe des ordonnées pour tout les mots de longueur k
    :param nom_seq: nom de la sequence qu'on teste
    :param sequence: une sequence d'ADN
    :param k: longueur des mots qu'on teste
    """
    freq = [0,0,0,0]
    tot = []
    nb_tot = 0
    seq_sans_titre=[]
    if type(sequence) is list:
        freq = Projet_Bioinfo.nucleotide_frequency(sequence)
        nb_tot = sum(Projet_Bioinfo.nucleotide_count(sequence))
        seq_sans_titre = sequence
        #print(freq, nb_tot)
    else:
        for i in sequence.keys():
            freq+= Projet_Bioinfo.nucleotide_frequency(sequence.get(i))
            tot += Projet_Bioinfo.nucleotide_count(sequence.get(i))
            seq_sans_titre += sequence.get(i)
        freq = freq/len(sequence.keys())
        nb_tot = sum(tot)
        #print(freq, nb_tot)

        
    mots = mots_possibles(k)
   #mots_lettres = mots_possibles_lettres(k)
    
    #le comptage attendu
    attendu = comptage_attendu(k,mots,freq,nb_tot)
    #le comptage obserbé dans la sequence
    observe = comptage_observe(k, seq_sans_titre, mots)
    #print(seq_sans_titre)


    x_values = attendu.values()
    y_values = observe.values()

    graphe = plt.subplot()
    graphe.plot(x_values, y_values, 'o')    # les mots
    
    #min et max des 2 axes
    limites = [np.min([graphe.get_xlim(), graphe.get_ylim()]), 
               np.max([graphe.get_xlim(), graphe.get_ylim()])]
    graphe.plot(limites, limites)   # une droite x=y
    
    graphe.set_xlim(limites)
    graphe.set_ylim(limites)
    graphe.set_xlabel("nombre d’occurrences attendu")
    graphe.set_ylabel("nombred d’occurrences observé")
    graphe.set_title("Mot de longueur " + str(k) + " pour la sequence " + nom_seq)
    
    #ajouter etiquette pour les mots??
    
    plt.show()




#-------------------------
#       3.2 Simulation de séquences aléatoires
#-------------------------
    
#----------Question 1-------------
def simule_sequence(lg, m):
    """
    :param lg: longueur de la sequence qu'on va simuler
    :param m: liste de taille 4 tq m[i] contient la fréquence du nucléotide
            représenté par i
    :return: génére une séquence aléatoire de longueur lg d'une composition 
    donnée m (proportion de A, C, G et T)
    """
    return np.random.choice(4, lg, p=m)


#----------Question 3-------------
def proba_empirique(mot, lg, m, N):
    """
    :param mot: mot representer sous une liste d'entier
    :param lg: longueur de la sequence qu'on va simuler
    :param m: liste de taille 4 tq m[i] contient la fréquence du nucléotide
            représenté par i
    :param N: nb total de simulation à faire
    :return: un dict tq dic[i] donne la probabilité que le mot m apparaisse 
            i fois dans une sequence de longueur lg
            si proba est nulle alors il n'apparait pas dans le dic
    
    """
    dic = dict()
    
    k = len(mot)
    mot_code = code(mot, k)
    for i in range(N):
        seq = simule_sequence(lg, m)
        compt = comptage_observe(k, seq, mots_possibles(k))
        if mot_code in compt:   # si cest dans le comptage des mots
            cpt = compt[mot_code]
        else:
            cpt = 0
        
        if cpt in dic:  # si cest deja dans le dico
            dic[cpt] += 1
        else :          # sinon on le creer
            dic[cpt] = 1
    
    
    return {cpt : dic[cpt]/N for cpt in dic}    #divise par N pour avoir la proba empirique




#-------------------------
#       3.3
#-------------------------
    
#----------Question 2-------------
    
def estimer_M(sequence):
    """
    :param sequence: séquence de nucléotides sous forme de liste d'entier
    :return: estime M à partir des comptages des mots de longueur 2
    """
    cpt = comptage_observe(2, sequence, mots_possibles(2))
    M = np.zeros((4,4))
    
    for key in cpt:
        let = decode(key,2)
        M[let[0],let[1]] = cpt[key]
        
    for i in range(4):
        M[i, :] /= M[i, :].sum()
    
    return M
    
#----------Question 3-------------
def simule_sequence_markov(f, M, l):
    """
    :param f: liste de 4 entiers représentant les fréquences des lettres
    :param M: mattrice de transition de la chaîne 
    :param l: longueur de la sequence qu'on veut simuler
    :return: une séquence de longueur l avec le modéle de dinucléotique
    """
    seq = list()
    seq.append(np.random.choice(4, p=f))    #1ere lettre en utilisant la frequence initial
    
    for i in range(1, l):
        seq.append(np.random.choice(4, p=M[seq[i-1], :]))
    return seq
    
#----------Question 4-------------
def proba_position(f, M, mot):
    """
    :param f: liste de 4 entiers représentant les fréquences des lettres
    :param M: mattrice de transition de la chaîne 
    :param mot: un mot sous forme d'une liste d'entier
    """
    res = f[mot[0]] #frequence de la premmiere lettre
    for i in range(1, len(mot)):
        res *= M[mot[i-1], mot[i]] 
    return res

#----------Question 5-------------
def comptage_attendu_markov(k, possibles, freq, nb, M):
    """
    :param k: taille des occurences qu'on cherche
    :param possibles: tableau de mots possibles de longueru k
    :param freq : tuple de 4 éléments contenant la frequence attendue de chaque lettre
    :param nb : nombre total de nucléotides dans la séquence
    :param M: matrice de transition
    :return: un dictionnaire associant les mots possibles et leur comptage attendu
    """
    dico = dict((key, 0) for key in possibles)
    for mot in possibles:
      proba = 1       
      mot_tab = decode(mot,k)
      proba *= proba_position(freq, M, list(mot_tab))
      dico[mot] = proba*int(nb)
    
    return dico


#----------Question 7-------------

def graphe_attendu_observe_markov(nom_seq, sequence, k, M):
    """
    Affiche un graphe avec le nombre attendu d’occurrences sur l’axe des abcisses 
    et le nombre observé sur l’axe des ordonnées pour tout les mots de longueur k
    avec le modele de markov
    :param nom_seq: nom de la sequence qu'on teste
    :param sequence: une sequence d'ADN
    :param k: longueur des mots qu'on teste
    :param M: matrice de transition
    """
    freq = [0,0,0,0]
    tot = []
    nb_tot = 0
    seq_sans_titre=[]
    if type(sequence) is list:
        freq = Projet_Bioinfo.nucleotide_frequency(sequence)
        nb_tot = sum(Projet_Bioinfo.nucleotide_count(sequence))
        seq_sans_titre = sequence
        #print(freq, nb_tot)
    else:
        for i in sequence.keys():
            freq+= Projet_Bioinfo.nucleotide_frequency(sequence.get(i))
            tot += Projet_Bioinfo.nucleotide_count(sequence.get(i))
            seq_sans_titre += sequence.get(i)
        freq = freq/len(sequence.keys())
        nb_tot = sum(tot)
        #print(freq, nb_tot)

        
    mots = mots_possibles(k)
   #mots_lettres = mots_possibles_lettres(k)
    
    #le comptage attendu
    attendu = comptage_attendu_markov(k,mots,freq,nb_tot, M)
    #le comptage obserbé dans la sequence
    observe = comptage_observe(k, seq_sans_titre, mots)
    #print(seq_sans_titre)


    x_values = attendu.values()
    y_values = observe.values()

    graphe = plt.subplot()
    graphe.plot(x_values, y_values, 'o')    # les mots
    
    #min et max des 2 axes
    limites = [np.min([graphe.get_xlim(), graphe.get_ylim()]), 
               np.max([graphe.get_xlim(), graphe.get_ylim()])]
    graphe.plot(limites, limites)   # une droite x=y
    
    graphe.set_xlim(limites)
    graphe.set_ylim(limites)
    graphe.set_xlabel("nombre d’occurrences attendu")
    graphe.set_ylabel("nombred d’occurrences observé")
    graphe.set_title("Mot de longueur " + str(k) + " pour la sequence " + nom_seq)
    
    #ajouter etiquette pour les mots??
    
    plt.show()



#-------------------------
#       3.4
#-------------------------
    
#----------Question 3-------------
def proba_empirique_markov(mot, lg, m, N, M):
    """
    :param mot: mot representer sous une liste d'entier
    :param lg: longueur de la sequence qu'on va simuler
    :param m: liste de taille 4 tq m[i] contient la fréquence du nucléotide
            représenté par i
    :param N: nb total de simulation à faire
    :param M: matrice de transition
    :return: un dict tq dic[i] donne la probabilité que le mot m apparaisse 
            i fois dans une sequence de longueur lg
            si proba est nulle alors il n'apparait pas dans le dic
    """
    dic = dict()
    k = len(mot)
    mot_code = code(mot, k)
    
    for i in range(N):
        seq = simule_sequence_markov(m,M, lg)
        compt = comptage_observe(k, seq, mots_possibles(k))
        if mot_code in compt:   # si cest dans le comptage des mots
            cpt = compt[mot_code]
        else:
            cpt = 0
        
        if cpt in dic:  # si cest deja dans le dico
            dic[cpt] += 1
        else :          # sinon on le creer
            dic[cpt] = 1
    return {cpt : dic[cpt]/N for cpt in dic}    #divise par N pour avoir la proba empirique


#----------Question 4-------------
    
    
    
    
    
    
    
    