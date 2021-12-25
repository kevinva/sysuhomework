#include <iostream>

using namespace std;

#define LENGTHP (sizeof(p)/sizeof(p[0]))
#define LENGTHQ (sizeof(q)/sizeof(q[0]))
#define DYNAMICDEBUG(TY, NM, N)                                           \
            printf("Showing type:" #TY ", variable name:" #NM "\n");      \
            showbst<TY>((NM),(N))


template <typename T>
void freebst(T **p, int n) {
    for (int i = 0; i < n; i++)
        delete[] p[i];
    delete[] p;
}

template <typename T>
void showbst(T **p, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            if ((float)p[i][j] != -1.0f)
                printf("%.2f ", (float)p[i][j]);
            else
                printf("%-05s", " ");
        printf("\n");
    }
    printf("\n");
}


int **optimalbst(float *p, float *q, int lengthp, int lengthq, float ***e, float ***w) {//返回root，记录着对应的索引
    if (lengthp <= 0 || lengthq <= 0) return NULL;
    int **root = new int *[lengthp], i, j;
    float t;
    *e = new float *[lengthq];
    *w = new float *[lengthq];
    for (i = 0; i < lengthp; i++)
        root[i] = new int[lengthp];
    for (i = 0; i < lengthq; i++) {
        (*e)[i] = new float[lengthq];
        (*w)[i] = new float[lengthq];
    }
    //初始化数据
    for (i = 0; i < lengthp; i++)
        for (j = 0; j < lengthp; j++)
            root[i][j] = -1;
    for (i = 0; i < lengthq; i++)
        for (j = 0; j < lengthq; j++)
            (*e)[i][j] = (*w)[i][j] = -1;
    //动态规划算法
    for (i = 0; i < lengthq; i++)
        (*e)[i][i] = (*w)[i][i] = q[i];
    for (int l = 1; l <= lengthp; l++)
        for (i = 1; i <= lengthp - l + 1; i++) {
            j = i + l - 1;
            (*w)[i - 1][j] = (*w)[i - 1][j - 1] + p[j - 1] + q[j];//i下标全部调整，j下标保持
            for (int r = i; r <= j; r++) {
                t = (*e)[i - 1][r - 1] + (*e)[r][j] + (*w)[i - 1][j];
                if ((*e)[i - 1][j] == -1.0f || t < (*e)[i - 1][j]) {
                    (*e)[i - 1][j] = t;
                    root[i - 1][j - 1] = r - 1;
                }
            }
        }
    return root;
}


void printfbst(int **root, int i, int j, int n) {
    if (i == 0 && j == n - 1)
        printf("k%d是根\n", root[i][j]+1);
    if (i < j)
    {
        int index = root[i][j];
        if (index != i)
            printf("k%d是k%d的左节点\n", root[i][index - 1] + 1, index + 1);
        printfbst(root, i, index - 1, n);
        if (index != j)
            printf("k%d是k%d的右节点\n", root[index + 1][j] + 1, index + 1);
        printfbst(root, index + 1, j, n);
    }
    else if (i == j)
    {
        printf("d%d是k%d的左节点\n", i, i + 1);
        printf("d%d是k%d的右节点\n", i + 1, i + 1);
    }
    else
        printf("d%d是k%d的右节点\n", j + 1, j + 1);
}


int main()
{
    system("chcp 65001"); // 解决window输出中文乱码问题
    
    float p[] = { 0.15,0.1,0.05,0.1,0.2 }, q[] = { 0.05,0.1,0.05,0.05,0.05,0.1 }, **e, **w;//k1 - k5, d0-d5
    int **root;//root保存的是数组下标
    root = optimalbst(p, q, LENGTHP, LENGTHQ, &e, &w);
    DYNAMICDEBUG(float, e,LENGTHQ);
    DYNAMICDEBUG(float, w, LENGTHQ);
    DYNAMICDEBUG(int, root, LENGTHP);
    printf("解释最优二叉树，数组下标转换为实际的k1-k5, d0-d5\n");
    printfbst(root, 0, LENGTHP-1, LENGTHP);
    freebst<int>(root, LENGTHP);
    freebst<float>(e, LENGTHQ);
    freebst<float>(w, LENGTHQ);

    

    return 0;
}