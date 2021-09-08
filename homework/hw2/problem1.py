'''
This program will simulate experiment stated
on problem 1.10 from LFD
@author Houhua Li
'''
import random as rd
import matplotlib.pyplot as plt

#rd.seed(10)
'''
Generate the 1000 coins, with front 1 and back 0
'''
# coins = [[1,0]]*1000
# print(coins)
'''
Each entry corresponds to a coin's flip record
'''
flip_record = []

for i in range(1000):
    flip = []
    # flip for 10 times independently
    for j in range(10):
        flip.append(rd.choice([1,0]))
    flip_record.append(flip)

#print(flip_record)
'''
Each entry is the number of head in 10 flips
of each coin
'''
flip_head = list(map(lambda x : x.count(1), flip_record))
#print(flip_head)

c_1 = flip_record[0]
c_rand = rd.choice(flip_record)
mini_head = min(flip_head)
c_min = None
for i in range(1000):
    if flip_head[i] == mini_head:
        c_min = flip_record[i]
        break

print(c_1, c_rand, c_min)

print(sum(flip_head))
