import numpy as np
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

# 获得Householder矩阵
def getHouseholderMatrix(vec):
    n = vec.size()[0]
    I = torch.eye(n)
    e1 = torch.zeros(n)
    e1[0] = 1
    v = vec + torch.norm(vec) * e1
    w = v / torch.norm(v)

    H = I - 2 * torch.matmul(w.view(-1, 1), w.view(1, -1))
    return H


def main(mat):
    n = mat.size()[0]
    eig_vec_list = []
    eig_val_list = []
    H_list = []
    r1_list = []
    evectmp_list = []
    for i in range(n):
        print(f'epoch: {i+1} ==========================================')
        eig_vec, eig_val = powerIteration(mat)
        eig_val_list.append(eig_val)

        if i == 0:
            eig_vec_list.append(eig_vec)
        else:
            evectmp_list.append(eig_vec)
            eval_idx = len(eig_val_list) - 1
            for H_mat, r1_vec, evectmp in zip(reversed(H_list), reversed(r1_list), reversed(evectmp_list)):
                alpha = r1_vec.dot(evectmp) / (eig_val_list[eval_idx] - eig_val_list[eval_idx - 1])
                alpha = alpha.unsqueeze(0)
                tmp_vec = torch.cat((alpha, evectmp)).view(-1, 1)
                final_vec = torch.matmul(H_mat, tmp_vec)

        H = getHouseholderMatrix(eig_vec) # 获得Householder矩阵
        A = torch.matmul(torch.matmul(H, mat), H.mT)  # 用Householder矩阵对原始矩阵做正交相似变换

        mat = A[1:, 1:]
        r1 = A[0, 1:]
        H_list.append(H)
        r1_list.append(r1)
        
        if mat.size()[0] == 1:
            eig_val_list.append(mat[0][0])
            break
    
    return torch.stack(eig_val_list).numpy(), eig_vec_list


if __name__ == '__main__':
    # mat = torch.rand((10, 10))
    mat = torch.tensor([[3, -1, 1], [2, 0, 1], [1, -1, 2]], dtype=torch.float32)
    print(f'mat: {mat}')
    eig_vals, eig_vecs = main(mat)
    print(f'eig_vals={eig_vals}')

    # test_evals, test_evecs = torch.eig(mat, eigenvectors=True)
    # print(f'test_evals: {test_evals}')
    # print(f'test_evects: {test_evecs}')

    test2_evals, test2_evecs = np.linalg.eig(mat)
    print(f'test2_evals: {test2_evals}')
    print(f'test2_evecs: {test2_evecs}')


