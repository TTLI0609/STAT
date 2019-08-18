import numpy as np
import random

import matplotlib.pyplot as plt
maxGreedy = 100
def jouer(listMachines, levier):
    """
    jouer un coup
    :param listMachines: liste de probabilité de gagner des N-1 levier
    :param levier: entier entre 0 et N-1 representant le levier
    :return: 1 si on a gagner, 0 sinon
    """

    n = random.uniform(0, 1)
    if n <= listMachines[levier]:
        return 1                #gagner
    else:
        return 0                #perdu


#test
#print(jouer([0.7 ,0.4 ,0.1], 0))
    

def aleatoire( moyen, n):
    """
    choisir aléatoirement un levier
    :param moyen: liste des récompences moyennes estimée pour tous les leviers à partir des essais du joueur
    :param n: liste des nombres de fois où le levier a été choisi
    :return: un entier representant le levier choisi
    """
    i = random.randint(0, len(n)-1)
    return i


#print(aleatoire([0.5,0.5,0.5], [2,2,3]))


def greedy(moyen, n ):
    """
    si on a déjà jouer plus de maxGreedy fois, on jouer le levier avec la récompense la plus élevée, sinon on joue aléatoirement
    :param moyen: liste des récompences moyennes estimée pour tous les leviers à partir des essais du joueur
    :param n: liste des nombres de fois où le levier a été choisi
    :return: un entier representant le levier choisi
    """
    if np.sum(n) > maxGreedy:
       return np.argmax(moyen)
    else:
       return aleatoire(moyen,n)


#print(greedy([0,0.5,0.2], [2,3,2,3]))
    

def epsilonGreedy(moyen,n):
    """ on choisit au hasard une probabilite de e pour expolrer le levier
    et il y a la probabilite de 1-e pour jouer le levier avec la recompense la plus elevee
    """
    e=random.uniform(0,1) #si e=0.1 pour tester
    if random.random()>e:
        return np.argmax(moyen)
    else:
        return aleatoire(moyen,n)
 
#print(epsilonGreedy([0,0.5,0.2,0.4], [2,3,2,1]))


def UCB(moyen,n):
    """on trouve un intervalle de confiance qui est la plus recompense
    """
    t = np.sum(n) 
    return np.argmax( (moyen)+np.sqrt(2*np.log(t)/  n[np.argmax(moyen) ]  ) ) 

#print(UCB([0.1,0.5,0.8,0.3], [0,1,1,3]))


def run(listMachines, algo, T):
    """
    jouer au bandits-manchots
    :param listMachines: liste de probabilité de gagner des N-1 levier
    :param algo: le nom de l'algo qu'on va utiliser
    :param T: nombre de fois qu'on va jouer
    :return: un tuple (gainDuJoueur, gainMaxTheorique)
    """

    moyen = [0 for i in range(len(listMachines))]
    n = [0 for i in range(len(listMachines))]
    listGain = [0 for i in range(len(listMachines))]

    gainJoueur = 0                        #gain obtenu par le joueur

    # calculer le gain max theorique
    maxU = np.max(listMachines)
    gainMax = T * maxU


    for j in range(0,T):
        i = algo(moyen,n)
        res = jouer(listMachines, i)
        n[i] += 1

        if res == 1:
            gainJoueur += 1
            listGain[i] += 1
            moyen[i] = listGain[i] / n[i]
        else:
            moyen[i] = listGain[i] / n[i]

    return gainJoueur, gainMax


#jouer :
maxGreedy = 100
T= 10000    #input("Jouer combien de fois ? ")
listMachines = [0.11 ,0.21, 0.33, 0.43, 0.212, 0.666, 0.1234,0.8,0.9]
"""
[0.11 ,0.21, 0.33, 0.43, 0.212, 0.666, 0.1234,0.8,0.9]


[0.1, 0.8]

[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
[0.11 ,0.21, 0.33, 0.43, 0.212, 0.666, 0.1234]
[0.11 ,0.21, 0.33, 0.43, 0.212, 0.666, 0.1234,0.8,0.9]
[0.8, 0.8, 0.8, 0.8, 0.8]
"""


print(run(listMachines, epsilonGreedy, T))


x = [i for i in range(0, T, 100)]        # x = temps T
y = []                                  # y = regret
for i in range(0, T, 100) :
    res = run(listMachines, epsilonGreedy, i)
    y.append(res[1] - res[0])

print(y)
plt.plot(x,y)
plt.show()

