import numpy as np
import matplotlib.pyplot as plt
import math

def func(x):
    return 54 * np.power(x, 6) + 45 * np.power(x, 5) - 102 * np.power(x, 4) - 69 * np.power(x, 3) + 35 * np.power(x, 2) + 16 * x - 4

def plot(f):
    """画函数图像"""
    x_list = np.linspace(-2, 2, 1000)
    y_list = [f(x) for x in x_list]
    plt.figure(figsize=(16, 9))
    plt.ylim((-30, 100))
    plt.plot(x_list, y_list)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def find_root(f, x_start, x_end, x_final=None):
    """求区间内的根"""
    x_result = 0
    x1 = x_start
    x2 = x_end
    iter_count = 0
    c_list = []
    while iter_count < 50:
        y1 = f(x1)
        y2 = f(x2)
        x_tmp = x2 - (y2 * (x2 - x1)) / (y2 - y1)
        x1 = x2
        x2 = x_tmp

        if x_final is not None:
            c_list.append(evaluate_convergence_speed(x_result, x_tmp, x_final))

        if math.fabs(x_tmp - x_result) < 1e-6:
            break
        x_result = x_tmp
        iter_count += 1
        # print(x_result)
        
    print(f'iter count={iter_count}, between=({x_start},{x_end}), x_result={x_result}')
    print(f'c_list = {c_list}')
    return x_result

def evaluate_convergence_speed(x1, x2, x_final):
    """验证收敛速度"""
    c = math.fabs(x2 - x_final) / math.fabs(x1 - x_final)
    return c

if __name__ == '__main__':
    plot(func)

    x1 = -2
    x2 = x1 + 0.04
    result_list = []
    while len(result_list) < 100 and x2 <= 2:
        x_result = find_root(func, x1, x2, x_final=-1.57366)
        result_list.append(x_result)
        
        x1 = x2
        x2 = x1 + 0.04

    [print(f'f(x)={func(result)}, x={result}') for result in result_list if math.fabs(func(result)) < 1e-3]