#include<iostream>
#include<fstream>
#include<time.h>
using namespace std;

//Gaming field
int field[3][3] = { {0, 0, 0},
					{0, 0, 0},
					{0, 0, 0} };
//Field for copying it every game
int origin_field[3][3] = {	{0, 0, 0},
							{0, 0, 0},
							{0, 0, 0} };


bool checkFinish() {	//������� ���������, �������� �� ����
	//���������, ���� �� ������ ����������
	for (int j = 0; j < 3; j++) {
		if ((field[j][0] == field[j][1]) and (field[j][0] == field[j][2]) and (field[j][1] == field[j][2]) and (field[j][0] != 0)) {
			return true;
		}
		if ((field[0][j] == field[1][j]) and (field[0][j] == field[2][j]) and (field[1][j] == field[2][j]) and (field[0][j] != 0)) {
			return true;
		}
	}

	if ((field[0][0] == field[1][1]) and (field[0][0] == field[2][2]) and (field[1][1] == field[2][2])  and (field[0][0] != 0)) {
		return true;
	}

	if ((field[0][2] == field[1][1]) and (field[0][2] == field[2][0]) and (field[1][1] == field[2][0]) and (field[0][2] != 0)) {
		return true;
	}
	//���� �� ����� ������, ������ � ��� ��� ��� ����������

	return false;
}

int checkWinner() {
	for (int j = 0; j < 3; j++) {
		if ((field[j][0] == field[j][1] == field[j][2]) and (field[j][0] != 0)) {
			return field[j][0];
		}
		if ((field[0][j] == field[1][j] == field[2][j]) and (field[0][j] != 0)) {
			return field[0][j];
		}
	}

	if ((field[0][0] == field[1][1] == field[2][2]) and (field[0][0] != 0)) {
		return field[0][0];
	}

	if ((field[0][2] == field[1][1] == field[2][0]) and (field[0][2] != 0)) {
		return field[0][2];
	}
}

class nn {
public:
	int layersN;
	int* arch;
	string* activationFunctions;
	long long neuronsNum = 0;
	long long weightsNum = 0;
	double* values;	//values of neurons
	double* errors;
	double* weights; // weights of neurons
	nn(int lN, int* Arch, string* AF) {
		layersN = lN;
		arch = new int[lN];
		activationFunctions = new string[lN];
		for (int i = 0; i < lN; i++) {
			arch[i] = Arch[i];
			activationFunctions[i] = AF[i];
			neuronsNum += arch[i];
			if (i > 0) {
				weightsNum += arch[i] * arch[i - 1];
			}
		}
		weights = new double[weightsNum];
		values = new double[neuronsNum];
		errors = new double[neuronsNum];
	}

	void gemm(int M, int N, int K, const double* A, const double* B, double* C)
	{
		for (int i = 0; i < M; ++i)
		{
			double* c = C + i * N;
			for (int j = 0; j < N; ++j)
				c[j] = 0;
			for (int k = 0; k < K; ++k)
			{
				const double* b = B + k * N;
				float a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += a * b[j];
			}
		}
	}
	void gemmSum(int M, int N, int K, const double* A, const double* B, double* C)
	{
		for (int i = 0; i < M; ++i)
		{
			double* c = C + i * N;
			for (int k = 0; k < K; ++k)
			{
				const double* b = B + k * N;
				float a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += a * b[j];
			}
		}
	}

	void New() {
		long long counter = 0;
		for (int i = 0; i < layersN - 1; i++) {
			for (int j = 0; j < arch[i] * arch[i + 1]; j++) {
				weights[counter] = (double(rand() % 201 - 100) / 100.0) / double(arch[i + 1]);
				counter++;
			}
		}
	}

	void act(int valuesC, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				values[i] = (1 / (1 + pow(2.71828, -values[i])));
			}

		}
		if (activationFunctions[layer] == "relu") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				if (values[i] < 0) values[i] *= 0.01;
				else values[i] = values[i];
			}
		}
		if (activationFunctions[layer] == "softmax") {
			double zn = 0.0;
			for (int i = 0; i < arch[layer]; i++) {
				zn += pow((2.71), values[valuesC + i]);
			}
			for (int i = 0; i < arch[layer]; i++) {
				values[valuesC + i] = pow(2.71, values[valuesC + i]) / zn;
			}
		}
	}

	void pro(double* value, int ecounter, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = 0; i < arch[layer]; i++) {
				values[ecounter + i] = values[ecounter + i] * (1.0 - values[ecounter + i]);
				value[i] *= values[ecounter + i];
			}
		}

		if (activationFunctions[layer] == "relu") {
			for (int i = 0; i < arch[layer]; i++) {
				if (values[i + ecounter] < 0) values[i] = 0.01;
				else values[i + ecounter] = 1;
				value[i] *= values[ecounter + i];
			}
		}

		if (activationFunctions[layer] == "softmax") {
			for (int i = 0; i < arch[layer]; i++) {
				value[i] *= values[ecounter + i] * (1.0 - values[ecounter + i]);
			}
		}
	}

	void ForwardFeed(double* input_data) {
		for (int i = 0; i < arch[0]; i++) {
			values[i] = input_data[i];
		}

		long long valuesC = 0;
		long long weightsC = 0;
		for (int i = 0; i < layersN - 1; i++) {
			double* a = values + valuesC;
			double* b = weights + weightsC;
			double* c = values + valuesC + arch[i];
			gemm(1, arch[i + 1], arch[i], a, b, c);


			//for (int j = valuesC; j < valuesC + arch[i + 1]; j++) {
			valuesC += arch[i];
			act(valuesC, i + 1);



			weightsC += arch[i] * arch[i + 1];

		}

	}

	void GetPrediction(double* result) {
		long long h = neuronsNum - arch[layersN - 1];
		for (int i = 0; i < arch[layersN - 1]; i++) {
			result[i] = values[h + i];
		}
	}

	void BackPropogation(double* rightResults, float lr) {
		//������� ���������� ������
		int h = neuronsNum - arch[layersN - 1];
		for (int i = 0; i < arch[layersN - 1]; i++) {
			errors[i + h] = rightResults[i] - values[i + h];
		}
		long long wcounter = weightsNum;
		long long ecounter = neuronsNum;
		long long counter = neuronsNum - arch[layersN - 1];
		for (int i = layersN - 1; i > 0; i--) {
			ecounter -= arch[i];
			wcounter -= arch[i] * arch[i - 1];
			counter -= arch[i - 1];
			double* a = errors + ecounter;
			double* b = weights + wcounter;
			double* c = errors + counter;
			gemm(1, arch[i - 1], arch[i], a, b, c);

		}

		//����� ���������� �����:
		long long vcounter = neuronsNum - arch[layersN - 1];
		wcounter = weightsNum;
		ecounter = neuronsNum;
		for (int i = layersN - 1; i > 0; i--) {
			ecounter -= arch[i];
			vcounter -= arch[i - 1];
			wcounter -= arch[i] * arch[i - 1];
			double* b = new double[arch[i]];
			for (int j = 0; j < arch[i]; j++) {
				b[j] = errors[ecounter + j] /*pro(values[ecounter + j], i)*/ * lr;
			}
			pro(b, ecounter, i);
			double* a = values + vcounter;
			double* c = weights + wcounter;

			gemmSum(arch[i - 1], arch[i], 1, a, b, c);

			delete[] b;
		}

	}

	void SaveWeights(string filename) {
		ofstream fout;
		fout.close();
		fout.open(filename);
		for (int i = 0; i < weightsNum; i++) {
			fout << weights[i] * 10000.0 << " ";
		}
		fout.close();
	}
};

int main() {

	srand(time(0));

	int gamesN = 10;	//���������� ���

	const int ln = 2;		//���������� ����� ���������
	int arch[ln] = { 9, 9 };	//����������� ���������
	string af[ln] = { "relu" };	//������������� ������� ���������
	nn player(ln, arch, af);	//������� "����" ������

	player.New();				//���������� ���� 

	for (int g = 0; g < gamesN; g++) {	//��� ������ ����
		//�������� ���� ����, �.�. ����� ����		
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				field[i][j] = origin_field[i][j];
			}
		}
		//����� 2 ������: 1 � 2,
		//����������, ��� �� ��� ����� ������
		int nowGoes = rand() % 2 + 1;

		for (int move = 0; (move < 9) and (!checkFinish()); move++) {

			//����, ���� ���� �� ����� ��������� ��� ����� ����������
			double input[9];	//������� ������ ��� ���������
			//��������� �� � ����������� �� ����, ����� ����� ����� 
			for (int i = 0; i < 9; i++) {
				if (nowGoes == 1) {
					if (field[i / 3][i % 3] == 0) input[i] = 0.0;
					if (field[i / 3][i % 3] == 1) input[i] = 1.0;
					if (field[i / 3][i % 3] == 2) input[i] = -1.0;
				}
				else {
					if (field[i / 3][i % 3] == 0) input[i] = 0.0;
					if (field[i / 3][i % 3] == 1) input[i] = -1.0;
					if (field[i / 3][i % 3] == 2) input[i] = 1.0;
				}
			}

			player.ForwardFeed(input);	//��������� ������ � ����

			double result[9];	//���� ����� �������� ��������� ���������

			player.GetPrediction(result);	//�������� ��������� �� ����

			//������ ���� ������������ �������� � ��� ������,
			//������� ����� �������, � ����� �������� ��������

			double maxValue = 0.0;
			int maxValueIndex = 0;

			for (int i = 0; i < 9; i++) {
				if (maxValue < result[i]) {
					maxValue = result[i];
					maxValueIndex = i;
				}
			}

			//������ ����� ���������, ������ �� ��� ��������

			if (field[maxValueIndex / 3][maxValueIndex % 3] != 0) {
				//���� ������� ������� ��������
				while (field[maxValueIndex / 3][maxValueIndex % 3] != 0) {
					//���� �������� ������� ��������, ��������� ���������
					//�� �������� ������� ��������

					//��� ����� ��������� ���������� ������
					double answers[9];
					for (int i = 0; i < 9; i++) {
						if (field[i / 3][i % 3] != 0) answers[i] = 0;
						else answers[i] = 1.0 / double(9 - move);
					}
					//������ ������� ��������� ������ �� ������ ��������
					player.BackPropogation(answers, 0.001);
					player.ForwardFeed(input);

					maxValue = 0.0;

					maxValueIndex = 0;

					for (int i = 0; i < 9; i++) {
						if (maxValue < result[i]) {
							maxValue = result[i];
							maxValueIndex = i;
						}
					}

				}
			}

			//���� �� ���, �� ������� ��������� ��������
			//������, �� ����� ������ ���
			
			field[maxValueIndex / 3][maxValueIndex % 3] = nowGoes;
			
			if (nowGoes == 1) nowGoes = 2;
			else nowGoes = 1;

			//����� ������� ���� ������ ������� ����
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					cout << field[i][j];
				}
				cout << endl;
			}

			cout << "Move: " << move << "\tCheckFinish: " << checkFinish() << endl;

		}
		
		cout << "Winner: " << checkWinner() << endl;

	}


	return 0;
}