import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random 

## Constante
OFFSET = 0.2


class State:
    """ Etat generique d'un jeu de plateau. Le plateau est represente par une matrice de taille NX,NY,
    le joueur courant par 1 ou -1. Une case a 0 correspond a une case libre.
    * next(self,coup) : fait jouer le joueur courant le coup.
    * get_actions(self) : renvoie les coups possibles
    * win(self) : rend 1 si le joueur 1 a gagne, -1 si le joueur 2 a gagne, 0 sinon
    * stop(self) : rend vrai si le jeu est fini.
    * fonction de hashage : renvoie un couple (matrice applatie des cases, joueur courant).
    """
    NX,NY = None,None
    def __init__(self,grid=None,courant=None):
        self.grid = copy.deepcopy(grid) if grid is not None else np.zeros((self.NX,self.NY),dtype="int")
        self.courant = courant or 1
    def next(self,coup):
        pass
    def get_actions(self):
        pass
    def win(self):
        pass
    def stop(self):
        pass
    @classmethod
    def fromHash(cls,hash):
        return cls(np.array([int(i)-1 for i in list(hash[0])],dtype="int").reshape((cls.NX,cls.NY)),hash[1])
    def hash(self):
        return ("".join(str(x+1) for x in self.grid.flat),self.courant)
            
class Jeu:
    """ Jeu generique, qui prend un etat initial et deux joueurs.
        run(self,draw,pause): permet de joueur une partie, avec ou sans affichage, avec une pause entre chaque coup. 
                Rend le joueur qui a gagne et log de la partie a la fin.
        replay(self,log): permet de rejouer un log
    """
    def __init__(self,init_state = None,j1=None,j2=None):
        self.joueurs = {1:j1,-1:j2}
        self.state = copy.deepcopy(init_state)
        self.log = None
    def run(self,draw=False,pause=0.5):
        log = []
        if draw:
            self.init_graph()
        while not self.state.stop():
            coup = self.joueurs[self.state.courant].get_action(self.state)
            log.append((self.state,coup))
            self.state = self.state.next(coup)
            if draw:
                self.draw(self.state.courant*-1,coup)
                plt.pause(pause)
        return self.state.win(),log
    def init_graph(self):
        self._dx,self._dy  = 1./self.state.NX,1./self.state.NY
        self.fig, self.ax = plt.subplots()
        for i in range(self.state.grid.shape[0]):
            for j in range(self.state.grid.shape[1]):
                self.ax.add_patch(patches.Rectangle((i*self._dx,j*self._dy),self._dx,self._dy,\
                        linewidth=1,fill=False,color="black"))
        plt.show(block=False)
    def draw(self,joueur,coup):
        color = "red" if joueur>0 else "blue"
        self.ax.add_patch(patches.Rectangle(((coup[0]+OFFSET)*self._dx,(coup[1]+OFFSET)*self._dy),\
                        self._dx*(1-2*OFFSET),self._dy*(1-2*OFFSET),linewidth=1,fill=True,color=color))
        plt.draw()
    def replay(self,log,pause=0.5):
        self.init_graph()
        for state,coup in log:
            self.draw(state.courant,coup)
            plt.pause(pause)

class MorpionState(State):
    """ Implementation d'un etat du jeu du Morpion. Grille de 3X3. 
    """
    NX,NY = 3,3
    def __init__(self,grid=None,courant=None):
        super(MorpionState,self).__init__(grid,courant)
    def next(self,coup):
        state =  MorpionState(self.grid,self.courant)
        state.grid[coup]=self.courant
        state.courant *=-1
        return state
    def get_actions(self):
        return list(zip(*np.where(self.grid==0)))
    def win(self):
        for i in [-1,1]:
            if ((i*self.grid.sum(0))).max()==3 or ((i*self.grid.sum(1))).max()==3 or ((i*self.grid)).trace().max()==3 or ((i*np.fliplr(self.grid))).trace().max()==3: return i
        return 0
    def stop(self):
        return self.win()!=0 or (self.grid==0).sum()==0
    def __repr__(self):
        return str(self.hash())

class Agent:
    """ Classe d'agent generique. Necessite une methode get_action qui renvoie l'action correspondant a l'etat du jeu state"""
    def __init__(self):
        pass
    def get_action(self,state):
        pass
    
    

class JoueurAlea(Agent):
    def __init__(self):
        pass
    def get_action(self,state):
        lenList = len(state.get_actions())
        action = random.randint(0,lenList-1)
        return state.get_actions()[action]
    


class MonteCarlo(Agent):
	def __init__(self):
		pass
	
	def get_action(self,state):
		lenList = len(state.get_actions())
		
		listRecom = [0] * lenList
		
		
		
		for i in range (0,100):
			j1=JoueurAlea()
			j2 =JoueurAlea()
			
			action = random.randint(0,lenList-1)
			
			state.next(state.get_actions()[action])
		
			jeu = Jeu(state, j1, j2)
			res = jeu.run(False, 0.1)
		
			if(jeu.state.win() == 1):
				listRecom[action] += 1
		
		ind = listRecom.index(max(listRecom))
		return state.get_actions()[ind]
				
				
class UCT(Agent):
	def __init__(self):
        pass
    def get_action(self,state):
        pass
    
         
               
               

"""
T = 1000  #nb de tours
j1 = JoueurAlea()
j2 = JoueurAlea()

def jouer():    #jouer un jeu de T partie avec 2 joueuer aleatoire
    listGagne = [0, 0, 0]   # nb de jeu gagné par 1, nb de jeu gagné par 2, égalité

    y = []                                  # y = nombre de parties gagné par joueur 1
    y2 = []
    y3 = []
    j = 0

    for i in range (0,T) :

        j+=1

        state = MorpionState(None, random.choice([-1, 1]))
        jeu1 = Jeu(state, j1, j2)
        jeu1.run(False, 0.5)

        if jeu1.state.win() == 1:
            listGagne[0] += 1
        if jeu1.state.win() == -1:
            listGagne[1] += 1
        if jeu1.state.win() == 0:
            listGagne[2] += 1

        if j == 10:
            j = 0
            y.append(listGagne[0])
            y2.append(listGagne[1])
            y3.append(listGagne[2])

    return y,y2,y3


#pour le graphe du aléatoire
x = [i for i in range(0, T, 10)]  # x = temps T nombre de jeu

l = jouer()
y = l[0]
y2 = l[1]
y3 = l[2]
for i in range(0,10):   # pour avoir la moyenne de 10 fois
    list2 = jouer()

    a = list2[0]
    b = list2[1]
    c = list2[2]

    y = [(y[i]+a[i])/2 for i in range(0, len(y))]
    y2 = [(y2[i] + b[i]) / 2 for i in range(0, len(y2))]
    y3 = [(y3[i] + c[i]) / 2 for i in range(0, len(y3))]

#print(y)
#print(y2)
#print(y3)

plt.plot(x,y,label="évolution du nombre de jeux gagné par J1")
plt.plot(x,y2, label="évolution du nombre de jeux gagné par J2")
plt.plot(x,y3, label=" égalité")
plt.legend()
plt.show()
"""




T = 100  #nb de tours
j1 = MonteCarlo()
j2 = JoueurAlea()

def jouer():    #jouer un jeu de T partie avec 2 joueuer aleatoire
    listGagne = [0, 0, 0]   # nb de jeu gagné par 1, nb de jeu gagné par 2, égalité

    y = []                                  # y = nombre de parties gagné par joueur 1
    y2 = []
    y3 = []
    j = 0

    for i in range (0,T) :

        j+=1

        state = MorpionState(None, random.choice([-1, 1]))
        jeu1 = Jeu(state, j1, j2)
        jeu1.run(False, 0.1)

        if jeu1.state.win() == 1:
            listGagne[0] += 1
        if jeu1.state.win() == -1:
            listGagne[1] += 1
        if jeu1.state.win() == 0:
            listGagne[2] += 1

        if j == 10:
            j = 0
            y.append(listGagne[0])
            y2.append(listGagne[1])
            y3.append(listGagne[2])

    return y,y2,y3
    
res = jouer()
print(res[0])
print(res[1])
print(res[2])
