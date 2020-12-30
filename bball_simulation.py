# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:15:48 2020

@author: harsh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sp = [0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600]
hitter = ["single","double","triple","home_run"]


class BaseState:
    def __init__(self, base1 = 0, base2=0, base3=0, score = 0):
        self.base1 = base1
        self.base2 = base2
        self.base3 = base3
        self.score = score
    
    def single(self):
        base_one = self.base1
        base_two = self.base2
        base_three = self.base3
        score = self.score
        
        if base_one == 1:
            self.base2 = 1
        if base_two == 1:
            self.base3 = 1
        if base_three == 1:
            self.score = score + 1  
        if base_one == 0:
            self.base2 = 0
        if base_two == 0:
            self.base3 = 0
        if base_three == 0:
            self.score = score
        self.base1 = 1
        
        print("Single Hitter: " + "\n" + "Base 1: {}".format(self.base1) + "\n" + "Base 2: {}".format(self.base2) + 
              "\n" + "Base 3: {}".format(self.base3) + "\n" + "Score: {}".format(self.score))
        
    def double(self):
        base_one = self.base1
        base_two = self.base2
        base_three = self.base3
        
        
        if base_one == 1:
            self.base3 = 1
            self.base1 = 0
        if base_two == 1:
            self.score = self.score + 1
        if base_three == 1:
            self.score = self.score + 1  
        if base_one == 0:
            self.base3 = 0
        if base_two == 0:
            self.score = self.score
        if base_three == 0:
            self.score = self.score
        self.base2 = 1
        
        print("Double Hitter: " + "\n" + "Base 1: {}".format(self.base1) + "\n" + "Base 2: {}".format(self.base2) + 
              "\n" + "Base 3: {}".format(self.base3) + "\n" + "Score: {}".format(self.score))
    
    def triple(self):
        base_one = self.base1
        base_two = self.base2
        base_three = self.base3
        
        
        if base_one == 1:
            self.score = self.score + 1
            self.base1 = 0
        if base_two == 1:
            self.score = self.score + 1
            self.base2 = 0
        if base_three == 1:
            self.score = self.score + 1  
        if base_one == 0:
            self.score = self.score
        if base_two == 0:
            self.score = self.score
        if base_three == 0:
            self.score = self.score
        self.base3 = 1
        
        print("Triple Hitter: " + "\n" + "Base 1: {}".format(self.base1) + "\n" + "Base 2: {}".format(self.base2) + 
              "\n" + "Base 3: {}".format(self.base3) + "\n" + "Score: {}".format(self.score))
     
    def home_run(self):
        base_one = self.base1
        base_two = self.base2
        base_three = self.base3
        
        
        if base_one == 1:
            self.score = self.score + 1
            self.base1 = 0
        if base_two == 1:
            self.score = self.score + 1
            self.base2 = 0
        if base_three == 1:
            self.score = self.score + 1  
            self.base3 = 0
        if base_one == 0:
            self.score = self.score
        if base_two == 0:
            self.score = self.score
        if base_three == 0:
            self.score = self.score
        self.score = self.score + 1
        
        print("Homer Hitter: " + "\n" + "Base 1: {}".format(self.base1) + "\n" + "Base 2: {}".format(self.base2) + 
              "\n" + "Base 3: {}".format(self.base3) + "\n" + "Score: {}".format(self.score))
    allFuncs = [single, double, triple, home_run]

def calculate_score(slugging_percentage): 
    batter = BaseState()
    result = []
    index = 1.0
    for type_of_hitter in batter.allFuncs: 
        batting_average = slugging_percentage/index
        avg_score = []
        for i in range(1000):
            batter = BaseState(0,0,0,0)
            inning = 0
            while(inning < 9):
                batter = BaseState(0,0,0,batter.score)
                no_of_outs = 0
                while(no_of_outs < 3):
                    hit_prob = np.random.uniform(0,1)
                    if hit_prob <= batting_average:
                        type_of_hitter(batter)
                    else:
                        no_of_outs += 1
                inning += 1
            avg_score.append(batter.score)
        result.append(np.mean(np.array(avg_score)))
        index += 1
    return result
    



df = pd.DataFrame(columns = ['single', 'double', 'triple', 'home_run'])
for k in sp:
    slugging_percentage = k
    df = df.append({'single': calculate_score(slugging_percentage)[0],
              'double': calculate_score(slugging_percentage)[1],
              'triple': calculate_score(slugging_percentage)[2],
              'home_run': calculate_score(slugging_percentage)[3]}, ignore_index=True)

plt.figure()
plt.plot(sp, df.iloc[:,0], '-', label = "Single")
plt.plot(sp, df.iloc[:,1], '-', label = "Double")
plt.plot(sp, df.iloc[:,2], '-', label = "Triple")
plt.plot(sp, df.iloc[:,3], '-', label = "Home Run")
plt.xlabel("Slugging Percentage")
plt.ylabel("Average Total Runs")
plt.title("Average Total Runs in 9 Inning Game (1,000 Sim)")
plt.legend()




#Combinations

combos = ["single_double", "single_triple", "double_triple", "single_homerun"]


def calculate_combo_score(slugging_percentage): 
    second_poss = [2.0,3.0,4.0,3.0] 
    first_poss = [1.0,1.0,1.0,2.0]
    batter = BaseState()         
    result = []
    for c in range(4):
        A = np.array([[1, -1], [first_poss[c], second_poss[c]]])
        B = np.array([0,slugging_percentage])
        X = np.linalg.inv(A).dot(B)
        batting_average = X[0]
        avg_score = []
        if c == 0:
            for i in range(1000):
                batter = BaseState(0,0,0,0)
                inning = 0
                while(inning < 9):
                    batter = BaseState(0,0,0,batter.score)
                    no_of_outs = 0
                    while(no_of_outs < 3):
                        hit_prob = np.random.uniform(0,1)
                        if hit_prob < batting_average:
                            batter.single()
                        elif hit_prob >= batting_average and hit_prob < 2*batting_average:
                            batter.double()
                        else:
                            no_of_outs += 1
                    inning += 1
                avg_score.append(batter.score)
            result.append(np.mean(np.array(avg_score)))
        elif c == 1:
            for i in range(1000):
                batter = BaseState(0,0,0,0)
                inning = 0
                while(inning < 9):
                    batter = BaseState(0,0,0,batter.score)
                    no_of_outs = 0
                    while(no_of_outs < 3):
                        hit_prob = np.random.uniform(0,1)
                        if hit_prob < batting_average:
                            batter.single()
                        elif hit_prob >= batting_average and hit_prob < 2*batting_average:
                            batter.triple()
                        else:
                            no_of_outs += 1
                    inning += 1
                avg_score.append(batter.score)
            result.append(np.mean(np.array(avg_score)))
        elif c == 2:
            for i in range(1000):
                batter = BaseState(0,0,0,0)
                inning = 0
                while(inning < 9):
                    batter = BaseState(0,0,0,batter.score)
                    no_of_outs = 0
                    while(no_of_outs < 3):
                        hit_prob = np.random.uniform(0,1)
                        if hit_prob < batting_average:
                            batter.single()
                        elif hit_prob >= batting_average and hit_prob < 2*batting_average:
                            batter.home_run()
                        else:
                            no_of_outs += 1
                    inning += 1
                avg_score.append(batter.score)
            result.append(np.mean(np.array(avg_score)))
        elif c == 3:
            for i in range(1000):
                batter = BaseState(0,0,0,0)
                inning = 0
                while(inning < 9):
                    batter = BaseState(0,0,0,batter.score)
                    no_of_outs = 0
                    while(no_of_outs < 3):
                        hit_prob = np.random.uniform(0,1)
                        if hit_prob < batting_average:
                            batter.double()
                        elif hit_prob >= batting_average and hit_prob < 2*batting_average:
                            batter.triple()
                        else:
                            no_of_outs += 1
                    inning += 1
                avg_score.append(batter.score)
            result.append(np.mean(np.array(avg_score)))
    return result



df = pd.DataFrame(columns = ['single_double', 'single_triple', 'single_home_run', 'double_triple'])
for k in sp:
    slugging_percentage = k
    df = df.append({'single_double': calculate_combo_score(slugging_percentage)[0],
              'single_triple': calculate_combo_score(slugging_percentage)[1],
              'single_home_run': calculate_combo_score(slugging_percentage)[2],
              'double_triple': calculate_combo_score(slugging_percentage)[3]}, ignore_index=True)
        
plt.figure()
plt.plot(sp, df.iloc[:,0], '-', label = "Single + Double")
plt.plot(sp, df.iloc[:,1], '-', label = "Single + Triple")
plt.plot(sp, df.iloc[:,2], '-', label = "Single + Home Run")
plt.plot(sp, df.iloc[:,3], '-', label = "Double + Triple")
plt.xlabel("Slugging Percentage")
plt.ylabel("Average Total Runs")
plt.title("Average Total Runs in 9 Inning Game (1,000 Sim)")
plt.legend()   
