import random

def randList(n, a=0,b=10):
	l=[]
	for i in range(n):
		l.append(random.randint(a,b))
	return l

#test
#print(randList(5,10,20))



def moyenne(l):
	res=0
	for i in l:
		res+=i
	return res / len(l)
	
#test
#print(moyenne([0,20,20,0]))


def histo(l):
	dic=dict()
	for i in l:
		if(i in dic):
			dic[i]+=1
		else:
			dic[i]=1
	return dic

#test
#print(histo([0,0,0,1,1,8,0,0,1]))

def histo_trie(l):
	liste=[]
	for i in l : 
		liste.append((l.count(i),i))
		
	liste=list(set(liste))
	sorted(liste,key=lambda liste: liste[0])
	return liste
	

	
#test
print(histo_trie([0,0,0,1,1,8,0,0,1]))


