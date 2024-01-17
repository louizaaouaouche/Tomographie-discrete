#___________________________________________________________________________#
#                           TOMOGRAPHIE DISCRETE                            #
#___________________________________________________________________________#
# LU3IN003: Algorithmique Avancé                                            #
# Auteurs: Louiza AOUAOUCHE et Ines YAICI                                   #
# Encadrants: Olivier SPANJAARD et Fanny PASCUAL                            #
#___________________________________________________________________________#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import time


#---------------------------------------------------------------------------#
#                               CONSTANTES GLOBALES                         #
#---------------------------------------------------------------------------#
VIDE = -1
NO_COLOR= 0
WHITE = 1
BLACK = 2

#constantes de choix de tests 
METHODE1=1
METHODE2=2

#---------------------------------------------------------------------------#
#                     LECTURE DU FICHIER /AFFICHAGE                         #
#---------------------------------------------------------------------------#

def ReadFile(fichier:str):
    """str -> list[list[int]]*list[int]*int*list[list[int]]*list[int]*int
    Lit le fichier F et retourne la grille associée et ses dimensions """

    #Ouverture du fichier
    try:
        f = open(fichier,"r")
    except IOError:
        print ("ReadFile2: Fichier <%s> introuvable, arret du programme"%(fichier))
        sys.exit(1) 
    
    
    #Lignes,Colonnes:list[list[int]] , stocke les séquences de chaque ligne (resp. colonnes)
    Lignes=[]
    Colonnes=[]

    # Li_Len,Col_Len:list[int] , stocke la taille des séquences de chaque ligne (resp. colonnes)
    Li_Len=[]
    Col_Len=[]
    
    #separateur:str , detecteur de passage aux colonnes dans le fichier F
    separateur=''
    
    #cpt,N,M:int
    cpt=0
    N=0
    M=0

    for ligne in f:
        if ligne=='#\n':
            separateur='#'
            N=cpt
            cpt=0
        if separateur != '#':

            
            ligne=ligne.split()
            L=list(map(int,ligne))
            Lignes.append(L)
            Li_Len.append(len(L))
            cpt=cpt+1
            
        else:
            if ligne!='#\n':
                
                ligne=ligne.split()
                L=list(map(int,ligne))
                Colonnes.append(L)
                Col_Len.append(len(L))
                cpt=cpt+1

    M=cpt    
    
    f.close()
    return Lignes,Li_Len,N,Colonnes,Col_Len,M

def Affichage(A:list,N:int,M:int,str_instance:str):
    '''A:list[list[int]]*N:int*M:int*str_instance:str->None
    Affiche la grille A de taille NxM (pas/partiellement/complètement coloriée)
    associée à l'instance: str_instance '''

    # Configuration des couleurs
    DRAW_WHITE=colors.to_rgb('white')
    DRAW_BLACK =colors.to_rgb('black')
    DRAW_GRIS=colors.to_rgb('gray')

    #Construction de la grille de dessin à partir de A

    A_draw=[]

    for i in range(N):
        A_draw.append([])
        for j in range(M):
            if A[i][j] == BLACK:
                A_draw[i].append(DRAW_BLACK)
            elif A[i][j] == WHITE:
                A_draw[i].append(DRAW_WHITE)
            else:
                A_draw[i].append(DRAW_GRIS)

    #Configuration des titres  
    plt.figure("Instance : "+str_instance)  
    plt.title("Instance : "+str_instance)

    im = plt.imshow(A_draw,extent = (0, M, 0, N)) 
    
    #Configuration des axes
    im.axes.tick_params(color = 'black', labelcolor="w", direction = 'out')
    axes = plt.gca()

    x=[i for i in range(M)]
    y=[j for j in range(N)]

    axes.set_xticks(x)
    plt.xlabel('M='+str(M))
    axes.set_yticks(y)
    plt.ylabel('N='+str(N))
    plt.savefig(str_instance+'.png')
    plt.show()

def Affiche_tout():
    """ Genère toutes les grilles des instances en testant Coloration"""

    for i in range (0,17):
        chaine=str(i)+".txt"
        N,M,Bool,A=Coloration(chaine)
        Affichage(A,N,M,chaine)

def Affiche_tout2():
    """ Genère toutes les grilles des instances en testant Enumération"""

    for i in range (0,17):
        chaine=str(i)+".txt"
        N,M,Bool,A=Enumeration(chaine)
        Affichage(A,N,M,chaine)


#---------------------------------------------------------------------------#
#                              1.1 PREMIERE PARTIE                          #
#---------------------------------------------------------------------------#

def T(t:list,s:list,j:int,l:int):
    ''' t:list[list[bool]]*s:list[int]*j:int *l:int -> bool 
    détecte s'il est possible de colorier une ligne jusqu'à
    j avec une sous séquencesous séquence de s qui va jusqu'à l. t est un tableau qui stocke
    les valeurs T(j,l) déjà calculées(ca s'applique sur les colonnes egalement)''' 

    if (t[j][l] == VIDE) :
        
        if l==0 :
            # la séquence est vide
            t[j][l] = True 
        
        if l>=1 :   
        # il y'a au moins un bloc dans la séquence
            if j < s[l]-1 : 
                # bloc plus grand que le nombre de cases
                t[j][l] = False
            if j == s[l]-1 :             
                if l==1 : 
                    # le 1er bloc est stockable dans les j+1 premières cases
                    t[j][l] = True   
                else :    
                    # il manque une case de séparation avec le bloc précédent
                    t[j][l] = False

            if j > s[l]-1 : 
                if j-s[l]-1 < 0 :
                    t[j][l] =T(t,s,j-1,l)
                else : 
                    # . j1:j-1 // si la case (i,j1) est coloriable c'est la dernière case noire
                    # du bloc l. Donc la case (i,j) est coloriable en blanc d'où T(t,s,j-1,l)
                    # . j2=j-s[l]-1 // si la case (i,j2) au bloc l-1 est coloriable, alors entre
                    # les cases (i,j2) et (i,j) il y'a suffisament de cases + un séparateur blanc
                    # pour que la case (i,j) soit noire (dernière case du bloc l)
                    t[j][l] =(T(t,s,j-s[l]-1,l-1)) or (T(t,s,j-1,l))

    return t[j][l]

#---------------------------------------------------------------------------#
#                              1.2 GENERALISATION                           #
#---------------------------------------------------------------------------#

def T_ge(t:list,Li:list, j:int, s:list, l:int):
    ''' t : list[list[bool]], Li: list[int], j : int, s : list[int], l : int -> bool
    Retourne True s'il est possible de colorier Li[:j] avec la séquence s
    en prenant compte des cases déjà coloriées stockées dans 't' '''
    
    if t[j][l] == VIDE:  # Si t[j][l] n'est pas vide on renvoie sa valeur déjà calculée

        ### CAS 1 ###

        # S'il n'y a pas de blocs 
        if l == 0:

            # Cette ligne doit être blanche donc si la ligne contient une case noire on renvoie False 
            t[j][l]=True
            for i in range (j):
                if Li[i]==BLACK:
                    t[j][l]=False
                    break
           
        ### CAS 2 ###

        # S'il y a au moins un bloc 
        else:

            ### CAS 2.A ###

            # on vérifie qu'on ne dépasse pas les j+1 cases
            if j < s[-1]:
                t[j][l] = False
            
            ### CAS 2.B ###

            # si les l blocs rentrent exactement dans les j+1 cases 
            elif j == s[-1]:
                if l==1:
                    #on vérifie si la ligne contient des cases blanches
                    t[j][l]=True
                    for i in range (j):
                        if Li[i]==WHITE:
                            t[j][l]=False
                            break
           
                else:
                    t[j][l]=False
            
            ### CAS 2.C ###

            # Si les l blocs peuvent rentrer largement dans les j+1 cases
            elif j > s[-1]:
               
                # Si la dernière case est blanche, on doit obligatoirement vérifier si la séquence rentre dans les j premières cases ( et non pas j+1)
                if Li[j-1] == WHITE:
                    t[j][l] = T_ge(t,Li, j-1, s, l)
                    
                
                elif Li[j-1]  == BLACK: # Si la dernière case est noire il faut que ce soit celle du dernier bloc s[l-1])

                    # il faut au moins une case blanche de séparation entre les blocs
                    if Li[j - s[-1] - 1] == BLACK:
                        t[j][l] = False
                        return t[j][l]

                    # On vérifie qu'aucune case devant être coloriée noire par le bloc n'est déjà blanche
                    if WHITE in Li[j-s[-1]:j]:
                        t[j][l] = False
                        return t[j][l]

                    # Sinon on vérifie si s[:l-1] rentre dans Li[:j-s[-1]-1]
                    t[j][l] = T_ge(t,Li, j-s[-1]-1, s[:-1], l-1)
                
                
                elif Li[j-1]  == NO_COLOR:# Si la dernière case n'est pas coloriée
                    
                    #BoolWhite: bool , qui vaut true si on n'a pas de case blanche entre j-s[-1] et j
                    BoolWhite=True
                    for i in range (j-s[-1],j):
                        if Li[i]==WHITE:
                            BoolWhite=False
                            break
    
                    if (Li[j - s[-1] - 1] != BLACK) and BoolWhite: 
                        # Si c'est le cas, elle peut être noire ou blanche
                        t[j][l] = T_ge(t,Li, j-1, s, l) or T_ge(t,Li, j-s[-1] - 1, s[:-1], l-1) #s[:-1]=(s1, ...,s_{l-1})
                    
                    else:
                        # Sinon la case est blanche
                        t[j][l] = T_ge(t,Li, j-1, s, l)

    return t[j][l]

#---------------------------------------------------------------------------#
#                              1.3 PROPAGATION                              #
#---------------------------------------------------------------------------#
def ColoreLig(grille:list, s:list,len_s:int,i:int,taille:int):
    '''Li : list[int],s:list[int],len_s:int,i:int,taille:int -> bool
    Renvoie True si Li peut être coloriée par la séquence s  '''

    #Li:list[list[int]] ,tableau relatif à la i-ème ligne 
    Li=grille[i]

    #t:list[list[bool]] , tableau de mémorisation (programmation dynamique)
    t=[]
    for j in range (taille +1):
        t.append([])
        for l in range (len_s +1):
            t[j].append(VIDE)
    
    return T_ge(t,Li,taille, s, len_s)

def ColoreCol(grille:list, s:list,len_s:int,j:int,taille:int):
    ''' : list[list[int]],s:list[int],len_s:int,j;int,taille:int -> bool
    Renvoie vrai si la j-ème colonne de grille peut être coloriée par la séquence s  '''

    #Cj:list[list[int]] ,tableau relatif à la j-ème colonne 
    Cj=[]
    for Li in grille:
        Cj.append(Li[j])

    #t:list[list[bool]] ,tableau de mémorisation (programmation dynamique)

    t=[]
    for j in range (taille +1):
        t.append([])
        for l in range (len_s +1):
            t[j].append(VIDE)
    
    return T_ge(t,Cj,taille, s, len_s)

def Coloration(fichier:str):
    '''File->int*int*bool*list[list[int]]
    Renvoie une grille coloriée si l'instance du fichier est coloriable, une grille vide sinon'''
    Lignes,Li_Len,N,Colonnes,Col_Len,M= ReadFile(fichier)
    
    #LignesAVoir,ColonnesAVoir:set , initialise les lignes (resp. colonnes) à traiter
    LignesAVoir=set([i for i in range(N)])
    ColonnesAVoir=set([j for j in range(M)])

    #grille:list[list[int]] , grille illustrant l'instance
    grille=[]
    for i in range(N):
        grille.append([])
        for j in range(M):
            grille[i].append(NO_COLOR)

    #Traitement des lignes et des colonnes

    while  LignesAVoir or ColonnesAVoir:
        
        for i in LignesAVoir:
            
            for j in range(M):
                if grille[i][j] == NO_COLOR:
                    
                    #----------- on teste le blanc -----------
                    grille[i][j] = WHITE # on colorie la case temporairement en blanc
                    ok_blanc = ColoreLig(grille, Lignes[i],Li_Len[i],i,M)
                    
                    #----------- on teste le noir -----------
                    grille[i][j] = BLACK# on colorie la case temporairement en noir
                    ok_noir = ColoreLig(grille,  Lignes[i],Li_Len[i],i,M)
                    grille[i][j] = NO_COLOR
                    # ********************* les deux tests échouent *********************
                    if (not ok_blanc) and (not ok_noir):
                        grille_vide=[[NO_COLOR for _ in range(M)] for _ in range(N)]
                        return (N,M,False,grille_vide)
                    # ********************* le test noir réussit ************************
                    if (not ok_blanc) and ok_noir:
                        grille[i][j] = BLACK #on colorie la case en noir
                        ColonnesAVoir.add(j)# ajout de j aux colonnes à voir prochainement
                       
                    # ********************* le test blanc réussit ***********************
                    if ok_blanc and (not ok_noir):
                        grille[i][j] = WHITE #on colorie la case en blanc
                        ColonnesAVoir.add(j)# ajout de j aux colonnes à voir prochainement
                        
        LignesAVoir = set()
        
        for j in ColonnesAVoir:
            for i in range(N):
                
                if grille[i][j] == NO_COLOR:
                   
                    #----------- on teste le blanc -----------
                    grille[i][j] = WHITE # on colorie la case temporairement en blanc
                    ok_blanc = ColoreCol(grille,  Colonnes[j],Col_Len[j],j,N)

                    #----------- on teste le noir -----------
                    grille[i][j] = BLACK# on colorie la case temporairement en noir 
                    ok_noir = ColoreCol(grille,Colonnes[j],Col_Len[j],j,N)
                    grille[i][j] = NO_COLOR
                    # ********************* les deux tests échouent *********************
                    if (not ok_blanc) and (not ok_noir):
                        grille_vide=[[NO_COLOR for _ in range(M)] for _ in range(N)]
                        return (N,M,False,grille_vide)

                    # ********************* le test noir réussit ************************
                    if (not ok_blanc) and ok_noir:
                        grille[i][j] = BLACK #on colorie la case en noir
                        LignesAVoir.add(i) # ajout de i aux lignes à voir prochainement

                    # ********************* le test blanc réussit ************************
                    if ok_blanc and (not ok_noir):
                        grille[i][j] = WHITE #on colorie la case en blanc
                        LignesAVoir.add(i) # ajout de i aux lignes à voir prochainement
                    
                        
        ColonnesAVoir = set()

    #Detection de cases non coloriées
    for i in range(N):
        for j in range(M):
            if grille[i][j] == NO_COLOR:
                return (N,M,None, grille)

    #Toutes les cases ont été coloriées 
    return (N,M,True, grille)

#---------------------------------------------------------------------------#
#                       2.Méthode complète de résolution                    #
#---------------------------------------------------------------------------#

def ColorierEtPropager(G:list,ii:int,jj:int,c:int,Lignes:list,Colonnes:list,Li_Len:list,Col_Len:list,N,M):
    """ list[list[int]]*int*int*int*list[int]*list[int]*list[int]*list[int]*int*int-> int*list[[]]
    A partir d'une instance dans le fichier F, on renvoie 0 s'il y'a impossibilité
    de coloration, 1 si toutes les cases ont été coloriées, 2 si on ne peut pas conclurer"""
      
    #grille:List[[int]], calque de G
    grille=[]
    #for Li in G:grille.append(Li)
    for i in range(N):
        grille.append([])
        for j in range(M):
            grille[i].append(G[i][j])


    #LignesAVoir,ColonnesAVoir:set , initialise les lignes (resp. colonnes) à traiter
    LignesAVoir={ii}
    ColonnesAVoir={jj}

    #coloration de la case (i,j) avec la couleur c
    grille[ii][jj]=c

    while  LignesAVoir or ColonnesAVoir:
        
        for i in LignesAVoir:
            
            for j in range(M):
                if grille[i][j] == NO_COLOR:
                    
                    #----------- on teste le blanc -----------
                    grille[i][j] = WHITE # on colorie la case temporairement en blanc
                    ok_blanc = ColoreLig(grille, Lignes[i],Li_Len[i],i,M)
                    
                    #----------- on teste le noir -----------
                    grille[i][j] = BLACK# on colorie la case temporairement en noir
                    ok_noir = ColoreLig(grille,  Lignes[i],Li_Len[i],i,M)
                    grille[i][j] = NO_COLOR
                    # ********************* les deux tests échouent *********************
                    if (ok_blanc==False) and (ok_noir==False):
                        grille_vide=[[NO_COLOR for _ in range(M)] for _ in range(N)]
                        return (False,grille_vide)
                    # ********************* le test noir réussit ************************
                    if (ok_blanc==False) and ok_noir:
                        grille[i][j] = BLACK #on colorie la case en noir
                        ColonnesAVoir.add(j)# ajout de j aux colonnes à voir prochainement
                       
                    # ********************* le test blanc réussit ***********************
                    if ok_blanc and (ok_noir==False):
                        grille[i][j] = WHITE #on colorie la case en blanc
                        ColonnesAVoir.add(j)# ajout de j aux colonnes à voir prochainement
                        
        LignesAVoir = set()
        
        for j in ColonnesAVoir:
            for i in range(N):
                
                if grille[i][j] == NO_COLOR:
                   
                    #----------- on teste le blanc -----------
                    grille[i][j] = WHITE # on colorie la case temporairement en blanc
                    ok_blanc = ColoreCol(grille,Colonnes[j],Col_Len[j],j,N)

                    #----------- on teste le noir -----------
                    grille[i][j] = BLACK# on colorie la case temporairement en noir 
                    ok_noir = ColoreCol(grille,Colonnes[j],Col_Len[j],j,N)
                    grille[i][j] = NO_COLOR
                    # ********************* les deux tests échouent *********************
                    if (ok_blanc==False) and (ok_noir==False):
                        grille_vide=[[NO_COLOR for _ in range(M)] for _ in range(N)]
                        return (False,grille_vide)
                    # ********************* le test noir réussit ************************
                    if (ok_blanc==False) and ok_noir:
                        grille[i][j] = BLACK #on colorie la case en noir
                        LignesAVoir.add(i) # ajout de i aux lignes à voir prochainement

                    # ********************* le test blanc réussit ************************
                    if ok_blanc and (ok_noir==False):
                        grille[i][j] = WHITE #on colorie la case en blanc
                        LignesAVoir.add(i) # ajout de i aux lignes à voir prochainement
                                    
        ColonnesAVoir = set()

    for i in range(N):
        for j in range(M):
            if grille[i][j] == NO_COLOR:
                return (None, grille)

    return (True, grille)    

def Enumeration(F:str):
    """ File -> int*int*bool*List[][]"""
    Lignes,Li_Len,N,Colonnes,Col_Len,M= ReadFile(F)
    _,_,ok,G=Coloration(F)
    
    if ok==False or ok==True:
        return (N,M,ok,G)
    

    (_,_,TESTWHITE,P)=Enum_Rec(G,0,WHITE,Lignes,Li_Len,N,Colonnes,Col_Len,M) 
    if TESTWHITE==True:
        return (N,M,TESTWHITE,P)
    
    return Enum_Rec(G,0,BLACK,Lignes,Li_Len,N,Colonnes,Col_Len,M)

def Enum_Rec(G:list,k:int,c:int,Lignes:list,Li_Len:list,N:int,Colonnes:list,Col_Len:list,M:int):
    """List[[int]]*int*int ->bool
    G: grille partiellement coloriée, k un indice de case, c une couleur"""

    if k==N*M :
        return (N,M,True,G) #toutes les cases sont coloriées

    i= k//M
    j= k%M
    #print("(i,j)=",i,j)

    ok,A=ColorierEtPropager(G,i,j,c,Lignes,Colonnes,Li_Len,Col_Len,N,M) 
    
    if ok!=None:
        return (N,M,ok,A)

    # indice de la prochaine case indeterminée à partir de k+1
    
    for k_prime in range(k+1,N*M):
        if G[k_prime//M][k_prime%M]==NO_COLOR:
            k=k_prime
            break

    #print(k)


    (_,_,TESTWHITE,P)=Enum_Rec(A,k,WHITE,Lignes,Li_Len,N,Colonnes,Col_Len,M)

    if TESTWHITE==True:
       
        return (N,M,TESTWHITE,P)
    
    return Enum_Rec(A,k,BLACK,Lignes,Li_Len,N,Colonnes,Col_Len,M) 

#---------------------------------------------------------------------------#
#                              TESTS                                        #
#---------------------------------------------------------------------------#
def Test(Num:int):
    """ Selon Num qui prend les constantes définies globalement cette fonction teste 
    la  partie 1, 2 ou 3"""

    if Num==METHODE1:
        TabTest=[]
        for i in range(0,4):
            TabTest.append([])
            for j in range(0,18):
                TabTest[i].append([])
        TabTest[0][0]="Instance"
        TabTest[1][0]="Nb Lignes"
        TabTest[2][0]="Nb Colonnes"
        TabTest[3][0]="Temps"
        N=0
        M=0
    
        for i in range (0,17):
            chaine=str(i)+".txt"
            TabTest[0][i+1]=chaine
            Lignes,Li_Len,N,Colonnes,Col_Len,M= ReadFile(chaine)
            startTime = time.time() #temps initial
            N,M,Bool,A=Coloration(chaine)
            endTime=time.time() 
            duree=endTime-startTime
            TabTest[1][i+1]=N
            TabTest[2][i+1]=M
            TabTest[3][i+1]=str(duree)
           
        for j in range (0,18):
            print(f'| {TabTest[0][j]:14} | {TabTest[1][j]:14} | {TabTest[2][j]:14} | {TabTest[3][j]:30} |'  )

        
    elif Num==METHODE2:
        TabTest=[]
        for i in range(0,4):
            TabTest.append([])
            for j in range(0,18):
                TabTest[i].append([])
        TabTest[0][0]="Instance"
        TabTest[1][0]="Nb Lignes"
        TabTest[2][0]="Nb Colonnes"
        TabTest[3][0]="Temps"
      
        N=0
        M=0
    
        for i in range (0,17):
            chaine=str(i)+".txt"
            TabTest[0][i+1]=chaine
            Lignes,Li_Len,N,Colonnes,Col_Len,M= ReadFile(chaine)
            startTime = time.time() #temps initial
            (N,M,Bool,A)=Enumeration(chaine)
            endTime=time.time() 
            duree=endTime-startTime
            TabTest[1][i+1]=N
            TabTest[2][i+1]=M
            TabTest[3][i+1]=str(duree)

        for j in range (0,18):
            print(f'| {TabTest[0][j]:14} | {TabTest[1][j]:14} | {TabTest[2][j]:14} | {TabTest[3][j]:30} |'  )

def MiniTest(chaine:str,f):
    """ teste la fonction f sur la chaine"""
    Lignes,Li_Len,N,Colonnes,Col_Len,M= ReadFile(chaine)
    startTime = time.time() #temps initial
    (N,M,Bool,A)=f(chaine)
    endTime=time.time() 
    duree=endTime-startTime
    print("OK,DUREE:",Bool,duree)
    Affichage(A,N,M,chaine)

##### TEST PARTIE 1 #### 
#Test(METHODE1)  
#Affiche_tout()

##### TEST PARTIE 1 ####
#Test(METHODE2)  
#Affiche_tout2()


##### MINI TEST #####

MiniTest('chien.txt',Coloration)
MiniTest('chien.txt',Enumeration)


