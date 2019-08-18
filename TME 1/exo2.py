import random
import matplotlib.pyplot as plt

def paquet():
	l=[]
	val=[i for i in range(1,14)]
	coul=['C','K','P','T']
	for i in range(len(val)):
		for j in range(len(coul)):
			l.append((val[i],coul[j]))
	random.shuffle(l)
	return l


#test
#print(paquet())


def meme_position(p,q):
	list=[]
	for i in range(len(p)):
		if(p[i]==q[i]):
			list.append(p[i])
	return list
			

#test
#print(meme_position(paquet(), paquet()))



def proba_meme_position():
	m=meme_position(paquet(),paquet())
	return len(m) / 52
	
#test
#print(proba_meme_position())




