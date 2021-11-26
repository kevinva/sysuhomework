#include<iostream>
#include<vector>
#include<string>
using namespace std;

vector<string> matrix_chain(string * p, int start, int length)
{
	vector<string> ret;
	if (length == 1) {
		ret.push_back(p[start]);
	}
	else if (length == 2) {
		ret.push_back("(" + p[start] + p[start + 1] + ")");
	}
	else {
		for (int i = 1; i != length; i++) {
			vector<string> part1 = matrix_chain(p, start, i);
			vector<string> part2 = matrix_chain(p, start + i, length - i);
			for (int k = 0; k != part1.size(); k++) {
				for (int m = 0; m != part2.size(); m++) {
					ret.push_back("(" + part1[k] + part2[m] + ")");
				}
			}
		}
	}
	return ret;
}

//n表示矩阵链的矩阵数目
int f(int n)
{
	int num=0;
	if (n == 1)
		num = 1;
	else
		for (int i = 1; i != n; i++)
			num += f(i)*f(n - i); //左子矩阵链完全加括号的数目乘以右子矩阵链完全加括号的数目
	return num;
}

int main()
{
	int n;
	cout << "请输入矩阵个数：";
	cin >> n;
	string* p = new string[n];  // 向量p用来存储n个矩阵
	for(int i=0; i!=n; i++)
	{
		p[i] = char(i + char('A'));
	}
	vector<string> result = matrix_chain(p, 0, n);
	cout << "理论上，方案数量：" << f(n) << endl;
	cout << "矩阵链乘完全括号化方案数量："<<result.size() << endl;
	for (int i = 0; i != result.size(); i++)
		cout << result[i] << endl;

	return 0;
}