import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculatePi(n: int):
    circle_count = 0
    
    for i in range(n):
        x = random.random()
        y = random.random()
        len_r = np.sqrt(np.power(x, 2) + np.power(y, 2))
        if len_r <= 1:
            circle_count += 1
            
    pi = circle_count * 4 / n
    return pi

def ex1():
    samples_list = [20, 50, 100, 200, 300, 500, 1000, 5000]
    categories = [str(x) for x in samples_list]
    epochs = 100
    result_dict = {}
    plt.figure(figsize=(16, 9))
    for index, n in enumerate(samples_list):
        result_list = []
        for c in range(epochs):
            result_list.append(calculatePi(n))
        result_dict[categories[index]] = result_list
        plt.plot(range(epochs), result_list)
    
    plt.legend(categories)
    plt.ylabel('Probability')
    plt.xlabel('epoch')
    plt.show()
    
    mean_list = [np.mean(v) for k, v in result_dict.items()]
    std_list = [np.var(v) for k, v in result_dict.items()]
    table = pd.DataFrame({'采用数目': samples_list, '均值': mean_list, '方差': std_list})
    print(table)
    
ex1()