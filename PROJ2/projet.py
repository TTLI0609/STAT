"""
Created on Fri Mar  1 16:01:30 2019
@author: 3679785
"""
import pandas as pd
import numpy as np
import math
import scipy
import utils
from matplotlib import pyplot

#-----------------------------------------
#Question 1 :
#-----------------------------------------
def getPrior(df):
    """
    Calculer la probabilité a priori de la classe 1 et l'intervalle de confiance à 
    95% pour l'estimation de cette probabilité
    :param df: Pandas Dataframe contenant les données. Il doit contenir une colonne nommée "target" 
               ne contenant que des 0 et 1.
    :return: Dictionnaire contennat la moyenne et les 2 extrémités de l'intervalle de confiance,
            Cléfs : 'estimation', 'min5pourcent', 'max5pourcent'
    """
    dic = dict()
    z = 1.960

    #moyenne
    dic['estimation'] = df['target'].mean()

    #intervalle de confiance
    tmp = z * math.sqrt( (dic['estimation'] * (1 -dic['estimation'])) / df.shape[0])
    dic['min5pourcent'] = dic['estimation'] - tmp
    dic['max5pourcent'] = dic['estimation'] + tmp
    return dic




#-----------------------------------------
#Question 2 :
#-----------------------------------------
class APrioriClassifier(utils.AbstractClassifier):
    """
    Estime très simplement la classe de chaque individu par la classe majoritaire
    """
    def __init__(self):
        pass
    def estimClass(self, attrs):
        """
        A partir des attributs, estime la classe 0 ou 1
        Pour APrioriClassifier, la classe est toujours 1
        
        :param attrs: dicionnaire nom-valeur des attributs
        :return: classe 0 ou 1 estimée
        """
        return 1

    def statsOnDF(self, df):
        """
        A partir d'un pandas.dataframe, calcule les taux d'erreurs de classification 
        et rend un dictionnaire.
        VP : nombre d'individus avec target=1 et classe prévue=1
        VN : nombre d'individus avec target=0 et classe prévue=0
        FP : nombre d'individus avec target=0 et classe prévue=1
        FN : nombre d'individus avec target=1 et classe prévue=0
        Précision : combien de candidats sélectionnés sont pertinents = VP/(VP+FP)
        Rappel : combien d'éléments pertinents sont sélectionnés  = VP/(VP+FN)
        
        :param df:  le dataframe à tester
        :return: un dictionnaire contennat les VP, FP, VN, FN, précision et rappel
        """
        dic = dict()
        dic['VP'] = 0
        dic['VN'] = 0
        dic['FP'] = 0
        dic['FN'] = 0

        for t in df.itertuples():
            dicEl=t._asdict()
            ligne = utils.getNthDict(df,dicEl.get('Index'))
            if( dicEl.get('target') == 1 and self.estimClass(ligne )==1):
                dic['VP'] += 1
            if (dicEl.get('target') == 0 and self.estimClass(ligne )== 0):
                dic['VN'] += 1
            if (dicEl.get('target') == 0 and self.estimClass(ligne)== 1):
                dic['FP'] += 1
            if (dicEl.get('target') == 1 and self.estimClass(ligne) == 0 ):
                dic['FN'] += 1

        dic['Précision'] = dic['VP']  /  (dic['VP']+dic['FP'])
        dic['Rappel'] = (dic['VP'])  / (dic['VP'] + dic['FN'])
        return dic



#-----------------------------------------
#Question 3 :
#-----------------------------------------
def P2D_l(df,attr):
    """
    Calcule la probabilité  P( attr | target )
    
    :param df: dataframe avec les données et doit contenir une colonne 'target'
    :param attr: nom d'une colonne dans le dataframe qu'on va utiliser
    :return: dictionnaire de dictionnaire tel que dic[t][a] = P(attr = a | target = t)
    """
    dic = dict()
    #les valeurs de "target" est soit 0 soit 1
    dic[1] = dict()
    dic[0] = dict()

    nbEl = df['target'].count()         #nombre total d'individu
    
    for t in df.itertuples():
        dicEl = t._asdict()

        probaAetB = df[(df[attr] == dicEl[attr]) & (df['target']== dicEl['target']) ][attr].count() /nbEl
        probaB =  df[df['target']== dicEl['target'] ]['target'].count()/ nbEl
        dic[dicEl['target']][dicEl[attr]] = probaAetB  / probaB
 
    return dic


def P2D_p(df,attr):
    """
    Calcule la probabilité P( target | attr )
    
    :param df: dataframe avec les données et doit contenir une colonne 'target'
    :param attr: nom d'une colonne dans le dataframe qu'on va utiliser
    :return: dictionnaire de dictionnaire tel que dic[a][t] = P( target = t | attr = a )
    """
    dic = dict()
    nbEl = df['target'].count()

    for t in df.itertuples():
        dicEl = t._asdict()
        probaAetB = df[(df[attr] == dicEl[attr]) & (df['target'] == dicEl['target'])][attr].count()/nbEl
        probaB = df[df[attr] == dicEl[attr]][attr].count() / nbEl
        if dicEl[attr] not in dic:
            dic[dicEl[attr]] = dict()
        dic[dicEl[attr]][dicEl['target']] = probaAetB / probaB
    return dic



class ML2DClassifier(APrioriClassifier):
    """
    utilise le principe du maximum de vraisemblance
    pour estimer la classe d'un individu à partir d'une seule colinne du dataframe
    """

    def __init__(self, df, attr):
        """
        Initialise un classifieur.
        table : apelle P2D_l pour crées le dictionnaire contennant 
        les probabilités P( attr | target )
        
        :param df: dataframe qui doit contenir une colonne "target"
        :param attr: le nom d'une colonne de df
        """
        self.attr = attr
        self.table = P2D_l(df, attr)

    def estimClass(self, attrs):
        """
        A partir des attributs, estime la classe 0 ou 1
        Pour ML2DClassifier, la classe est estimée par maximum de vraisemblance 
        à partir de l'attribut table
        
        :param attrs: dicionnaire nom-valeur des attributs
        :return: classe 0 ou 1 estimée
        """
        val0 = self.table[0]
        val0 = val0[attrs[self.attr]]
        val1 = self.table[1]
        val1 = val1[attrs[self.attr]]

        if val0 >= val1:
            return 0
        else:
            return 1


class MAP2DClassifier(APrioriClassifier):
    """
    utilise le principe du maximum a posteriori
    pour estimer la classe d'un individu à partir d'une seule colinne du dataframe
    """
    def __init__(self, df, attr):
        """
        Initialise un classifieur.
        table : apelle P2D_p pour crées le dictionnaire contennant 
        les probabilités P( target | attr)
        
        :param df: dataframe qui doit contenir une colonne "target"
        :param attr: le nom d'une colonne de df
        """
        self.attr = attr
        self.table = P2D_p(df, attr)

    def estimClass(self, attrs):
        """
        A partir des attributs, estime la classe 0 ou 1
        Pour MAP2DClassifier, la classe est estimée par maximum de vraisemblance 
        à partir de l'attribut table
        
        :param attrs: dicionnaire nom-valeur des attributs
        :return: classe 0 ou 1 estimée
        """
        tmp = self.table[attrs[self.attr]]
        val0 = tmp.get(0)
        val1 = tmp.get(1)

        if val0 >= val1:
            return 0
        else:
            return 1



#-----------------------------------------
#Question 4 :
#-----------------------------------------
def convertir(string, nb):
    """
    converti un nombre d'octet en str et con concatene avec string
    :param string: un string 
    :param nb : un nombre d'octet
    :return: un string 
    """
    k = 1024
    m = k * k
    g = m * k
    t = g * k
    
    if k <= nb <= m :
        q = int(nb / k)
        r = nb % (k*q)
        string += " = " + str(q) +"ko " + str(r) + "o"
    elif m < nb <= g:
        q = int(nb / m)
        r = nb % (m * q)
        nb = r
        string += " = " + str(q) + "mo "
        q = int(nb / k)
        r = nb % (k * q)
        nb = r
        string += " = " + str(q) + "ko " + str(r) + "o"
    elif g < nb <= t :
        q = int(nb / g)
        r = nb %(g * q)
        nb = r
        string += " = " + str(q) + "go "
        q = int(nb / m)
        r = nb % (m * q)
        nb = r
        string += str(q) + "mo "
        q = int(nb / k)
        r = nb % (k * q)
        nb = r
        string += str(q) + "ko " + str(r) + "o"
    elif nb > t:
        q = int(nb / t)
        r = nb %(t * q)
        nb = r
        string += " = " + str(q) + "to "
        q = int(nb / g)
        r = nb %(g * q)
        nb = r
        string +=  str(q) + "go "
        q = int(nb / m)
        r = nb % (m * q)
        nb = r
        string += str(q) + "mo "
        q = int(nb / k)
        r = nb % (k * q)
        nb = r
        string += str(q) + "ko " + str(r) + "o"
    else :
        pass
    return string


def nbParams(df, listPar=[]):
    """
    calcule la taille mémoire des tables P(target|attr_1,..,attr_k) 
    étant donné un dataframe et une liste [target,attr_1,...,attr_l]
    On supposant qu'1 float est représenté sur 8 octets
    
    :param df: un dataframe
    :param listPar: [target,attr_1,...,attr_l]
    :return: affiche une chaine de caractere : nous informant sur le nombre de variables
    et la taille mémoire totale utilisé
    """
    nb = 8
    res = list()
    
    if(len(listPar) == 0):
        listPar = list(df)
        
    taille = len(listPar)
     
    for i in listPar:
        tmp = df[i].drop_duplicates()
        res.append(tmp.count())
    
    for i in res:
        nb *= i
    
    #pour affichage 
    string = str(taille) + " variable(s)" + ": " + str(nb) + " octets"
    print(convertir(string, nb))
    


def nbParamsIndep(df):
    """
    calcule la taille mémoire des tables P(target|attr_1,..,attr_k) 
    étant donné un dataframe contenant les attribut [target,attr_1,...,attr_l]
    On supposant qu'1 float est représenté sur 8 octets, de plus on suppose qu'il y a
    une indépendance totale entre les attributs
    
    :param df: un dataframe
    et la taille mémoire totale utilisé
    """
    nb = 0
    res = list()

    taille = len(df.columns)
    listPar = list(df)
    
    for i in listPar:
        tmp = df[i].drop_duplicates()
        res.append(tmp.count())
   
    for i in res:
        nb += i * 8
    
    string = str(taille) + " variable(s)" + ": " + str(nb) + " octets"
    print(convertir(string,nb))
    
    
    
    

#-----------------------------------------
#Question 5 :
#-----------------------------------------
def drawNaiveBayes(df, a):
    string = ""
    for i in df.columns:
        if i != a:
            string += a+"->"+i+";"
    return utils.drawGraph(string)



def nbParamsNaiveBayes(df, a, list=None):
    val = 1
    nb = 0
    res = []

    if list == None:
        list = df.columns

    tmp = df[a].drop_duplicates()
    val = tmp.count()
    if (len(list) == 0):
        print("0 variable(s) : " + str(val*8) + " octets")
        return

    for i in list:
        if i == a :
            continue
        tmp = df[i].drop_duplicates()
        res.append(tmp.count())

    for i in res:
        nb += val * i * 8
    val = (val*8) + nb

    #affichage
    string = str(len(list)) + " variable(s)" + " : " + str(val) + " octets"
    print(convertir(string, val))




class MLNaiveBayesClassifier(APrioriClassifier):
    """
    utilise le maximum de vraisemblance (ML)  
    pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes.
    """
    def __init__(self, df):
        """
        Initialise un classifieur MLNaiveBayesClassifier.
        Utilise P2D_l pour calculer les proba.
        :param df: dataframe qui doit contenir une colonne "target"
        """
        self.probas = {attr: P2D_l(df, attr) for attr in list(df)}
        self.df = df

    def estimProbas(self, attrs):
        """
        Calcule la probabilité P( a1, a2,...|target )
        
        :param df: dataframe avec les données et doit contenir une colonne 'target'
        :param attr: nom d'une colonne dans le dataframe qu'on va utiliser
        :return: dictionnaire de clefs: 1 et 0 et pour valeur leur probabilité
        """
        d = {0: 1, 1: 1}
        for i in self.df.keys():
            if i != 'target':
                if attrs[i] in self.probas[i][0].keys():    #si cest dans probas[i][0] 
                    d[0] *= self.probas[i][0][attrs[i]]
                else:                                       #sinon cest nul
                    d[0] = 0  
                    
                if  attrs[i] in self.probas[i][1].keys():   #si cest dans probas[i][0] 
                    d[1] *= self.probas[i][1][attrs[i]]
                else:
                    d[1] = 0                                #sinon cest nul
                    
                if d[0]==0 and d[1]==0 :                    #si les 2 valeurs sont nulles alors on sort de la boucle
                    break
        return d

    def estimClass(self, attrs):
        """
        A partir des attributs, estime la classe 0 ou 1
        Pour MLNaiveBayesClassifier, la classe est estimée par maximum de vraisemblance 
        en utilisant l'hypothèse du Naïve Bayes.
        
        :param attrs: dicionnaire nom-valeur des attributs
        :return: classe 0 ou 1 estimée
        """
        d = self.estimProbas(attrs)
        if d[0] >= d[1]:
            return 0
        else:
            return 1


class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    utilise  le maximum a posteriori (MAP)
    pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes.
    """
    def __init__(self, df):
        """
        Initialise un classifieur MLNaiveBayesClassifier.
        Utilise P2D_l pour calculer les proba.
        :param df: dataframe qui doit contenir une colonne "target"
        """
        self.probas = {attr: P2D_l(df, attr) for attr in list(df)}
        self.df = df

    def estimProbas(self, attrs):
        """
        Calcule la probabilité P( target|a1, a2,... )
        
        :param df: dataframe avec les données et doit contenir une colonne 'target'
        :param attr: nom d'une colonne dans le dataframe qu'on va utiliser
        :return: dictionnaire de clefs: 1 et 0 et pour valeur leur probabilité
        """
        d = {0: 1, 1: 1}
        p1 = self.df.target.sum() / float(self.df.target.count())
        p0 = 1 - p1
        for i in self.df.keys():
            if (i != 'target'):
                if attrs[i] in self.probas[i][0].keys():    #si cest dans probas[i][0] 
                    d[0] *= self.probas[i][0][attrs[i]]
                else:                                       #sinon cest nul
                    d[0] = 0  
                    
                if  attrs[i] in self.probas[i][1].keys():   #si cest dans probas[i][0] 
                    d[1] *= self.probas[i][1][attrs[i]]
                else:
                    d[1] = 0                                #sinon cest nul
                    
                if d[0]==0 and d[1]==0 :                    #si les 2 valeurs sont nulles alors on sort de la boucle
                    break
        pAttrs = d[0] * p0 + d[1] * p1
        if pAttrs == 0:
            return d
        d[0] = d[0] * p0 / pAttrs
        d[1] = d[1] * p1 / pAttrs
        return d

    def estimClass(self, attrs):
        """
        A partir des attributs, estime la classe 0 ou 1
        Pour MAPNaiveBayesClassifier, la classe est estimée par maximum de vraisemblance 
        en utilisant l'hypothèse du Naïve Bayes.
        
        :param attrs: dicionnaire nom-valeur des attributs
        :return: classe 0 ou 1 estimée
        """
        d = self.estimProbas(attrs)
        if d[0] >= d[1]:
            return 0
        else:
            return 1




#-----------------------------------------
#Question 6 :
#-----------------------------------------   
def isIndepFromTarget(df,attr,x): 
    """
    vérifie si attr est indépendant de target au seuil de x%
    :param df: un dataframe
    :param attr: l'attribut dans df qu'on veut tester
    :param x: le seuil en %
    :return: true si cest independant, false sinon
    """
    res = []
    target_values = np.unique(df['target'].values)  
    attr_values = np.unique(df[attr].values)
    
    for i in target_values:
        res.append([])
        for j in attr_values:       #parcour tout les valeurs possible de attr
            res[i].append( df[ (df['target'] == i) & (df[attr]== j) ][attr].count())
    
    tmp = scipy.stats.chi2_contingency(res)
    if  tmp[1] < x:
        return False
    else:
        return True


            
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    utilise le maximum de vraisemblance (ML)  
    pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes,
    qu'ils auront préalablement optimisé grâce à des tests d'indépendance au seuil de x%
    """
    def __init__(self, df,x):
        """
        Initialise un classifieur ReducedMLNaiveBayesClassifier.
        On supprime les arguments qui sont independant avec un seuil de x%
        Utilise P2D_l pour calculer les proba.
        :param df: dataframe qui doit contenir une colonne "target"
        :param x: seuil de x%
        """
        self.df = df.copy()
        for i in df.columns:
            if( isIndepFromTarget(df,i,x) == True):
                del self.df[i]
        self.probas = {attr: P2D_l(self.df, attr) for attr in list(self.df)}
    
    def draw(self):
        """
        dessine les dependances des arguments 
        """
        return( drawNaiveBayes(self.df, 'target'))
            
            
class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    utilise  le maximum a posteriori (MAP)  
    pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes,
    qu'ils auront préalablement optimisé grâce à des tests d'indépendance au seuil de x%
    """
    def __init__(self, df,x):
        """
        Initialise un classifieur ReducedMAPNaiveBayesClassifier.
        On supprime les arguments qui sont independant avec un seuil de x%
        Utilise P2D_l pour calculer les proba.
        :param df: dataframe qui doit contenir une colonne "target"
        :param x: seuil de x%
        """
        self.df = df.copy() 
        for i in df.columns:
            if( isIndepFromTarget(df,i,x) == True):
                del self.df[i]
        self.probas = {attr: P2D_l(self.df, attr) for attr in list(self.df)}
    
    def draw(self):
        """
        dessine les dependances des arguments 
        """
        return( drawNaiveBayes(self.df, 'target'))
        
        
        

#-----------------------------------------
#Question 7 :
#-----------------------------------------
def mapClassifiers(dic,df):
    x=[]
    y=[]
    nom=[]
    for key,val in dic.items():
        nom.append(key)
        cl = val                    #le classifieur
        tmp = cl.statsOnDF(df)
        x.append(tmp['Précision'])
        y.append(tmp['Rappel'])
    
    fig, ax = pyplot.subplots()
    ax.scatter(x, y, c = 'red',marker ='x')

    for i, n in enumerate(nom):
        ax.annotate(n, (x[i], y[i]))
    
    pyplot.show()
    
    
    

    
    
    
    
    
    
    
    

