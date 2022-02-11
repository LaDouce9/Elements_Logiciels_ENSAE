# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:40:08 2021

@author: tdoucet
"""
import numpy as np
import pandas as pd
import math
import logging
import threading
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import multiprocessing as mp

def extract_coord(geostring : str):
    where=geostring.find(',')
    return[float(geostring[:where]),float(geostring[where+1:])]

def distance2(lat1,lon1,lat2,lon2):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    lat1 : float
        latitude of the original point
    lon1 : float
        longitude of the original point
    lat2 : float
        latitude of the point of destination
    lon2 : float
        longitude of the point of destination
    Returns
    -------
    distance_in_km : float
    """
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def CalculDistance_proc(q,df_orig : pd.DataFrame,df_dest : pd.DataFrame):
    """
    Function launched on an unique processor

    Parameters
    ----------
    df_orig : DataFrame
        DataFrame containing at least columns 'latitude' and 'longitude'
    df_dest : DataFrame
        DataFrame containing at least columns 'latitude' and 'longitude'
    Returns
    -------
    """
    res=pd.DataFrame(index=df_orig.index,columns=df_dest.index)
    start=time.time()
    for i in res.index:
        lat1=df_orig.loc[i,'latitude']
        lon1=df_orig.loc[i,'longitude']
        for j in res.columns:
            lat2=df_dest.loc[j,'latitude']
            lon2=df_dest.loc[j,'longitude']
            res.loc[i,j]=distance2(lat1,lon1,lat2,lon2)
    Letime=time.time()-start
    q.put(np.array([res,Letime], dtype=object))


def Distance_MultiProcess(df_origin :pd.DataFrame, df_destination :pd.DataFrame, nProc =2):
    """
    Function which splits the data and organize the processor according to the number nProc

    Parameters
    ----------
    df_orig : DataFrame
        DataFrame containing at least columns 'latitude' and 'longitude'
    df_dest : DataFrame
        DataFrame containing at least columns 'latitude' and 'longitude'
    Returns
    array df_final
        (df_final : DataFrame, exec_time: list)
    -------
    """
    ctx = mp.get_context('spawn')
    N = df_origin.shape[0]
    exec_time=0
    exec_time_process=[]
    time_communication=0
    Sorti=[]
    index_a = [N // nProc * i for i in range(nProc)]
    index_b = [N // nProc * (i + 1) for i in range(nProc)]
    if N % nProc != 0:
        index_b[-1]=N
    process = list()
    q_list = list()
    start = time.time()
    for p in range(nProc):
        q = ctx.Queue()
        q_list.append(q)
        logging.info("Main    : create and start process %d.", p)
        x = ctx.Process(target=CalculDistance_proc,
                        args=(q, df_origin.iloc[index_a[p]:index_b[p], :], df_destination))
        process.append(x)
        logging.info("start" + str(p))
        x.start()
    for q in q_list:
        temp = q.get()
        Sorti.append(temp[0])
        exec_time_process.append(temp[1])
        logging.info(str(temp[1]))
    for p, proc in enumerate(process):
        logging.info("Main    : before joining process %d.", p)
        proc.join()
        proc.close() # fermeture du processeur
        logging.info("Main    : process %d done", p)
    exec_time=time.time()-start #On stock le temps d'éxécution
    for j in range(len(Sorti)):
       if j==0:
           df_final=Sorti[j]
       else:
           df_final=pd.concat([df_final,Sorti[j]],ignore_index=True)
    time_communication=time.time()-start-exec_time
    return np.array([df_final,exec_time_process,exec_time,time_communication], dtype=object)


if __name__=='__main__':
    path_paul='C:/Users/petit/OneDrive/Bureau/Paul/MS/S1/elements logiciels/Elements_Logiciels_ENSAE-main/Elements_Logiciels_ENSAE-main/'
    data=pd.read_csv('annonces_immo.csv')
    data=data[['approximate_latitude','approximate_longitude']]
    data.columns = ['latitude', 'longitude']

    
    df_gare=pd.read_csv('referentiel-gares-voyageurs.csv',sep=';')
    #Suppression des 11 gares sans coordonnéées GPS
    df_gare=df_gare[df_gare['WGS 84'].isnull()== False]
    
    gareratp=pd.read_csv('emplacement-des-gares-idf.csv',sep=';')
    gareratp["latitude"]=gareratp.apply(
        lambda x: extract_coord(x['Geo Point'])[0],
        axis=1)
    gareratp["longitude"]=gareratp.apply(
        lambda x: extract_coord(x['Geo Point'])[1],
        axis=1)
    
    df_gare2=df_gare[['Latitude','Longitude','WGS 84']]
    gareratp2=gareratp[['latitude','longitude','Geo Point']]
    df_gare2.columns=gareratp2.columns

    #concatenation des DataFrame SNCF & RATP
    df_train=pd.concat([df_gare2,gareratp2],ignore_index=True)
    df_train.drop_duplicates(inplace=True) #Suppresion des gares en doublon en IDF
    

    elem_max = df_train.shape[0]  #Limiter taille
    df_dest2 = df_train[:elem_max].copy()
    
    elem_max2=data.shape[0] #Limiter taille
    df_origin2 = data[:elem_max2].copy()
    
    #nb_process=[1,2,3,4,5,6,7,8]
    nb_process=[1,2,4,8,16,32]
    time_process=[]
    time_exec=[]
    time_com=[]
   
    for nb_proc in nb_process:
        print("Multiprocess starting for {} Process".format(int(nb_proc)))
        t=time.time()
        res = Distance_MultiProcess(df_origin2, df_dest2, nProc=nb_proc)
        #Calcul distance la plus proche  pour  chaque annonce  puis  export
        pd.DataFrame(res[0].transpose().min()).to_excel('Res_Process/MultiProcess_N_'+str(nb_proc)+'.xlsx')
        time_process.append(res[1])
        time_exec.append(res[2])
        time_com.append(res[3])  #Temps pour récupérer les données et les merges
        t2=time.time()-t
        del res #Libérer de la mémoire
        print("Temps d'éxécution pour gares (Process {}: {} secondes".format(int(nb_proc),round(t2, 2)))

    

    c=2
    if len(time_process)%2==0:
        l=len(time_process)//2
    else:
        l=len(time_process)//2+1

    fig=plt.figure(figsize=(10,10))
    for i in range(len(time_process)):
        sub_arg=int(str(l)+str(c)+str(i+1))
        plt.subplot(sub_arg)
        plt.bar(range(1,int(nb_process[i])+1),time_process[i])
        plt.title('Temps execution par process pour {} process'.format(int(nb_process[i])))
        plt.xlabel('processeurs')
        if nb_process[i]<20:
            plt.xticks(range(1,int(nb_process[i])+1))
        plt.ylabel('time (s)')
    fig.tight_layout()
    plt.savefig('time_per_process.jpg',dpi=300)
    plt.show()
    
    plt.plot(range(1,len(nb_process)+1),time_exec)
    plt.xlabel('Nombre  de Processeurs')
    plt.ylabel('time (s)')
    plt.title("Temps  d'execution en fonction du nombre de processeurs")
    plt.savefig('time_process.jpg',dpi=300)
    plt.show()
    
    #on récupère le temps de communication :  soit le temps total -  le temps du processeur le plus lent 
    time_com=[]
    for i in range(len(time_process)):
        max_time_process=np.max(time_process[i])
        time_com.append(time_exec[i]-max_time_process)
        
    fig, ax1=plt.subplots()
    plt.plot(nb_process,time_exec,label='Temps total éxécution')
    plt.ylabel('time (s) - Execution')
    plt.title('Temps en fonction du nombre de processeurs')
    plt.legend(loc='best')
    ax2=ax1.twinx()
    plt.plot(nb_process,time_com,'m--',label='Temps de communication')
    plt.xlabel('processeurs')
    plt.ylabel('time (s) - Communication')
    plt.legend(loc='best')
    plt.savefig('time_process_com.jpg',dpi=300)
    plt.show()

    print('Stop')


    
    
