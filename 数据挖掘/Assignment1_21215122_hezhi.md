### Assignment 1: Exercises for Monte Carlo Methods


* Lectured by 梁上松, Sun Yat-sen University
* Student ID:   21215122          
* Student Name: 何峙

#### Exercise 1

设圆半径为r，则：
$$\frac{4分圆面积}{正方形面积} = \frac{\frac{1}{4}\pi r^2 }{r^2} = \frac{\pi}{4}=\frac{落在4分圆的点数}{落在正方形点数}$$
即：
$$\pi=\frac{4 * 落在4分圆的点数}{落在正方形点数}$$

实验结果如下：
![./a2.jpg](./a2.jpg)

以下为个采样100个点的分布情况：
![./a1.png](./a1.png)


#### Exercise 2

概率约为47%， 方差约0.113

代码如下：
```
# ex2

def canMoveTo(x: int, y: int, grid: np.array):
    result = True
    mid = (grid.shape[0] // 2, grid.shape[0] // 2)
    if (x, y) == mid:
        if grid[(x, y)] >= 2:
            result = False
    else:
        if grid[(x, y)] >= 1:
            result = False
    return result


def moveInGridSize(size: int):
    assert size > 0
    mid = (size // 2, size // 2)
    
    grid = np.zeros((size, size), dtype=np.int32)
    current = (0, 0)
    grid[current] = 1
    path = [current]
    
    fail_count = 0
    
    while True:
        p = random.random()
        
        x = current[0]
        y = current[1]
        if current == (0, 0):
            if not canMoveTo(x + 1, y, grid) and not canMoveTo(x, y + 1, grid):
                # 绝路
                grid = np.zeros((size, size), dtype=np.int32)
                current = (0, 0)
                grid[current] = 1
                path = [current]
                fail_count += 1
                continue
            else:
                if p < 0.5:
                    x += 1 # 纵走
                else: 
                    y += 1 # 横走

            
        elif current == (0, size - 1):
            if not canMoveTo(x + 1, y, grid) and not canMoveTo(x, y - 1, grid):
                # 绝路
                grid = np.zeros((size, size), dtype=np.int32)
                current = (0, 0)
                grid[current] = 1
                path = [current]
                fail_count += 1
                continue
            else:                
                if p < 0.5: 
                    x += 1
                else:
                    y -= 1
        elif current == (size - 1, 0):
            if not canMoveTo(x - 1, y, grid) and not canMoveTo(x, y + 1, grid):
                # 绝路
                grid = np.zeros((size, size), dtype=np.int32)
                current = (0, 0)
                grid[current] = 1
                path = [current]
                fail_count += 1
                continue
            else:
                if p < 0.5:
                    x -= 1
                else:
                    y += 1
        elif current == (size - 1, size - 1):
                # 到达，结束！
#                 print('Finish!')
                break
        else:
            if x == 0:
                if not canMoveTo(x, y - 1, grid) and not canMoveTo(x, y + 1, grid) and not canMoveTo(x + 1, y, grid):
                    # 绝路
                    grid = np.zeros((size, size), dtype=np.int32)
                    current = (0, 0)
                    grid[current] = 1
                    path = [current]
                    fail_count += 1
                    continue
                else:
                    if p < 0.3333:
                        y -=1
                    elif 0.3333 <= p < 0.6667:
                        y += 1
                    else:
                        x += 1
            elif x == size - 1:
                if not canMoveTo(x, y - 1, grid) and not canMoveTo(x, y + 1, grid) and not canMoveTo(x - 1, y, grid):
                    # 绝路
                    grid = np.zeros((size, size), dtype=np.int32)
                    current = (0, 0)
                    grid[current] = 1
                    path = [current]
                    fail_count += 1
                    continue
                else:
                    if p < 0.3333:
                        y -=1
                    elif 0.3333 <= p < 0.6667:
                        y += 1
                    else:
                        x -= 1
            elif y == 0:
                if not canMoveTo(x - 1, y, grid) and not canMoveTo(x + 1, y, grid) and not canMoveTo(x, y + 1, grid):
                    # 绝路
                    grid = np.zeros((size, size), dtype=np.int32)
                    current = (0, 0)
                    grid[current] = 1
                    path = [current]
                    fail_count += 1
                    continue
                else:
                    if p < 0.3333:
                        x -=1
                    elif 0.3333 <= p < 0.6667:
                        x += 1
                    else:
                        y += 1
            elif y == size - 1:
                if not canMoveTo(x - 1, y, grid) and not canMoveTo(x + 1, y, grid) and not canMoveTo(x, y - 1, grid):
                    # 绝路
                    grid = np.zeros((size, size), dtype=np.int32)
                    current = (0, 0)
                    grid[current] = 1
                    path = [current]
                    fail_count += 1
                    continue
                else:
                    if p < 0.3333:
                        x -=1
                    elif 0.3333 <= p < 0.6667:
                        x += 1
                    else:
                        y -= 1
            else:
                if not canMoveTo(x - 1, y, grid) and not canMoveTo(x + 1, y, grid) and not canMoveTo(x, y - 1, grid) and not canMoveTo(x, y + 1, grid):
                    # 绝路
                    grid = np.zeros((size, size), dtype=np.int32)
                    current = (0, 0)
                    grid[current] = 1
                    path = [current]
                    fail_count += 1
                    continue
                else:
                    if p < 0.25:
                        x -= 1
                    elif 0.25 <= p < 0.5:
                        x += 1
                    elif 0.5 <= p < 0.75:
                        y -= 1
                    else:
                        y += 1
        
        if canMoveTo(x, y, grid):            
            path.append((x, y))
            current = (x, y)
            grid[current] += 1
    
#     print('grid: \n', grid)
#     print('path: ', path)
#     print('try count: ', fail_count)
    return fail_count

def ex2():
    p_list = []
    for i in range(20000):
        count = moveInGridSize(7) + 1 # 最后成功达到的尝试次数
        p_list.append(1.0 / count)
    
#     plt.figure(figsize=(16, 9))
#     plt.plot(range(len(p_list)), p_list)
#     plt.show()
#     print(p_list)
    print('mean: ', np.mean(p_list))
    print('var: ', np.var(p_list))
    
ex2()
```

#### Exercise 3

验证得概率约为97.8%
代码如下：
```
# ex3

def isASucceed():
    result = False
    p = random.random()
    if p <= 0.85:
        result = True
    return result

def isBCSucceed():
    result = False
    p1 = random.random()
    if p1 <= 0.95:
        p2 = random.random()
        if p2 <= 0.9:
            result = True
    return result


def ex3():
    total_count = 60000
    fail_count = 0
    for i in range(total_count):
        if not isASucceed() and not isBCSucceed():
            fail_count += 1
    
    print(1 - fail_count / total_count)
    
ex3()
```