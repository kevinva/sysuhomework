from numpy import power
import torch

N_MAX = 50
VAL_ERR = 1e-5

# 用幂法求矩阵m的最大特征值及对应特征向量
def powerIteration(m):
    x = torch.rand((m.size()[0],))
    n = 0
    err = float('inf')
    x_m = torch.max(x)
    eig_vec = torch.zeros((m.size()[0],))
    eig_val = 0
    while n < N_MAX and err >= VAL_ERR:
        x_u = x / x_m
        x = torch.matmul(m, x_u)

        eig_val = torch.max(x)
        eig_vec = x_u
        err = abs(eig_val - x_m)
        
        x_m = eig_val
        n += 1
        print(f'n: {n}, eig_vec: {eig_vec}, eig_val: {eig_val}')

    return eig_vec, eig_val


def getHouseholderMatrix(vec):
    n = vec.size()[0]
    I = torch.eye(n)
    e1 = torch.zeros(n)
    e1[0] = 1
    v = vec + torch.norm(vec) * e1
    w = v / torch.norm(v)

    H = I - 2 * torch.matmul(w.view(-1, 1), w.view(1, -1))
    return H


def getAll(mat):
    n = mat.size()[0]
    eig_vec_list = []
    eig_val_list = []
    eig_vec, eig_val = powerIteration(mat)
    eig_vec_list.append(eig_vec)
    eig_val_list.append(eig_val)
    H = getHouseholderMatrix(eig_vec)
    A = torch.matmul(torch.matmul(H, mat), H.mT)   # 用Householder矩阵对原始矩阵做正交相似变换
    for i in range(n - 1):
        r1 = A[0, 1:]
        mat = A[1:, 1:]



        eig_vec, eig_val = powerIteration(mat)
        eig_vec_list.append(eig_vec)
        eig_val_list.append(eig_val)
        H = getHouseholderMatrix(eig_vec)
        A = torch.matmul(torch.matmul(H, mat), H.mT)  



if __name__ == '__main__':
    mat = torch.rand((3, 3))
    # mat = torch.tensor([[3, -1, 1], [2, 0, 1], [1, -1, 2]], dtype=torch.float32)
    print(f'mat: {mat}')
    eig_vec, eig_val = powerIteration(mat)

    evals, evecs = torch.eig(mat, eigenvectors=True)
    print(evals)
    print(evecs)

    H = getHouseholderMatrix(eig_vec)
    A = torch.matmul(torch.matmul(H, mat), H.mT)
    print(A)
    print(A[0, 1:])
    print(A[2:, 2:])
