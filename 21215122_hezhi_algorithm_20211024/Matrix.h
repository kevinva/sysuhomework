#ifndef MATRIX_H
#define MATRIX_H

class  Matrix
{
public:
    Matrix();         //构造函数
    ~Matrix();        //析构函数
    bool Run();       //运行接口函数
    
private:
    int W;         //记录矩阵的个数
    int **m;       //存放最优值，即最小运算量
    int **s;       //断开位置
    int *p;        //存放

    bool Input();  //处理输入
    bool MatrixChain();//计算最优值算法
    void Traceback(int i,int j,int **s);   //输出矩阵加括号的方式
};

#endif