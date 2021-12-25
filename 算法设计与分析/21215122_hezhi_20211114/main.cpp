#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
using namespace std;
 
template<int N>
class NQueen
{
public:
    typedef array<int, N> ColumnType;
 
    NQueen()
    {
        Column.fill(-1);
    }
 
    /*使用回溯法，求出在一个n*n的棋盘上放置n个皇后，使其不能互相攻击的所有可能位置。 */
    void Nqueen()
    {
        int count = 0;
        int currentRow = 0; // currentRow 是当前行
 
        while(currentRow > -1) // 对所有的行执行以下语句
        { 
            Column[currentRow] += 1; //移动到下一列
            while((Column[currentRow] < N) &&
                (!Place(Column, currentRow)))
            {
                Column[currentRow] += 1;
            }
            if (Column[currentRow] < N) // 找到一个位置
            { 
                if(currentRow == (N-1)) // 是一个完整的解吗
                {
                    resVec.push_back(Column);
                }
                else
                {
                    currentRow++;
                    Column[currentRow] = -1; // 转向下一行
                } 
            }
            else
            {
                currentRow--; // 回溯
            }
        }
    }
 
    void PrintAll()
    {
        cout << "一共有" << resVec.size() << "组解。" << endl;
 
        for_each(resVec.begin(), resVec.end(), []
                 (ColumnType& column)
                 {
                     for_each(column.begin(), column.end(), [](int& value){cout << value << " ";});
                     cout << endl;
                 });
    }
 
private:
    vector<ColumnType> resVec; //存放所有的解
    ColumnType Column; //存放回溯过程中的一组成功解，下标代表行，数组值代表列
 
    //一个皇后是否能放在第 row 行，和第 Column[row] 列？
    bool Place(ColumnType& columnArray, int row)
    {
        for(int oldRow = 0; oldRow < row; ++oldRow)
        {
            if (columnArray[oldRow] == columnArray[row] || //同一行有两个皇后
                abs(columnArray[oldRow]-columnArray[row]) == abs(oldRow-row)) //在同一斜线上
            {
                return false;
            }
        }
        return true;
    }
};
 
int main()
{
    system("chcp 65001"); // 解决window输出中文乱码问题
    
    /* N皇后问题，找出在一个N*N的棋盘上防止N个皇后，并使其不能互相攻击的所有方案。 
    即，所有的皇后不能同行或同列。*/
    NQueen<8> Queen8;
 
    Queen8.Nqueen();
    Queen8.PrintAll();
}