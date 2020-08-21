#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
using namespace std;

//Gaming field
int field[3][3] = { {0, 0, 0},
					{0, 0, 0},
					{0, 0, 0} };
//Field for copying it every game
int origin_field[3][3] = {	{0, 0, 0},
							{0, 0, 0},
							{0, 0, 0} };


bool checkFinish() {	//Функция проверяет, окончена ли игра
	//проверяем, есть ли сейчас победитель
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
	//Если мы дошли досюда, значит у нас еще нет победителя

	return false;
}

int checkWinner() {
	for (int j = 0; j < 3; j++) {
		if ((field[j][0] == field[j][1]) and (field[j][0] == field[j][2]) and (field[j][1] == field[j][2]) and (field[j][0] != 0)) {
			return field[j][0];
		}
		if ((field[0][j] == field[1][j]) and (field[0][j] == field[2][j]) and (field[1][j] == field[2][j]) and (field[0][j] != 0)) {
			return field[0][j];
		}
	}

	if ((field[0][0] == field[1][1]) and (field[0][0] == field[2][2]) and (field[1][1] == field[2][2]) and (field[0][0] != 0)) {
		return field[0][0];
	}

	if ((field[0][2] == field[1][1]) and (field[0][2] == field[2][0]) and (field[1][1] == field[2][0]) and (field[0][2] != 0)) {
		return field[0][2];
	}
	return 0;
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
				weights[counter] = (double(rand() % 101) / 100.0) / double(arch[i + 1]);
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
				else values[i] = values[i] * 0.1;
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
				else values[i + ecounter] = 0.1;
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
		//Сначала вычисление ошибок
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

		//Потом обновление весов:
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

struct fieldCopy {
	int field[3][3];
};

int main() {

	srand(time(0));

	int gamesN = pow(10, 9);	//Количество игр

	const int ln = 3;		//Количество слоев нейросети
	int arch[ln] = { 9, 81, 9 };	//Архитектура нейросети
	string af[ln] = { "-", "relu", "softmax" };	//Активационные функции нейросети
	nn player(ln, arch, af);	//Создаем "мозг" игрока

	player.New();				//Генерируем веса


	//В такие векторы мы сохраняем состояние поля, при котором походил игрок
	//для дальнейшего обучения ИИ

	vector<fieldCopy> player1Fields;
	vector<fieldCopy> player2Fields;

	//В такие векторы мы сохраняем ход игрока для определенного состояния поля

	vector<int> player1Moves;
	vector<int> player2Moves;


	for (int g = 0; g < gamesN; g++) {	//Для каждой игры

		//Очищаем fields и moves
		player1Fields.clear();
		player2Fields.clear();

		player1Moves.clear();
		player2Moves.clear();


		int wrongChoice = 0;	//счетчик того, сколько раз ИИ выбрал занятую клеточку

		//Копируем наше поле, т.к. новая игра		
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				field[i][j] = origin_field[i][j];
			}
		}
		//Имеем 2 игрока: 1 и 2,
		//Определяем, кто из них ходит сейчас
		int nowGoes = rand() % 2 + 1;

		for (int move = 0; (move < 9) and (!checkFinish()); move++) {

			//Идем, пока поле не будет заполнено или будет победитель
			double input[9];	//Входные данные для нейросети
			//Заполняем их в зависимости от того, какой игрок ходит 
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

			player.ForwardFeed(input);	//Загружаем данные в сеть

			double result[9];	//Сюда будем получать результат нейросети

			player.GetPrediction(result);	//Получаем резульаты от сети

			//Теперь ищем максимальное значение и его индекс,
			//который будет значить, в какую клеточку походить

			double maxValue = 0.0;
			int maxValueIndex = 0;

			for (int i = 0; i < 9; i++) {
				if (maxValue < result[i]) {
					maxValue = result[i];
					maxValueIndex = i;
				}
			}

			//Теперь нужно проверить, занята ли эта клеточка

			if (field[maxValueIndex / 3][maxValueIndex % 3] != 0) {
				//Если выбрали занятую клеточку
				while (field[maxValueIndex / 3][maxValueIndex % 3] != 0) {

					//cout << "Wrong choice!\n";
					wrongChoice++;

					//Пока выбираем занятую клеточку, тренируем нейросеть
					//не выбирать занятые клеточки

					//Для этого указываем правильные ответы
					double answers[9];
					for (int i = 0; i < 9; i++) {
						if (field[i / 3][i % 3] != 0) answers[i] = 0.0;
						else answers[i] = 1.0 / double(9.0 - move);
					}


					//Теперь обучаем нейросеть ходить на пустые клеточки
					player.BackPropogation(answers, 0.1);
					player.ForwardFeed(input);
					player.GetPrediction(result);

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

			//Добавляем текущее состояние поля к вектору полей игрока
			//и добавляем его ход к вектору ходов
			if (nowGoes == 1) {
				fieldCopy copy;
				for (int i = 0; i < 9; i++) copy.field[i / 3][i % 3] = field[i / 3][i % 3];
				player1Fields.push_back(copy);
				player1Moves.push_back(maxValueIndex);
			}
			if (nowGoes == 2) {
				fieldCopy copy;
				for (int i = 0; i < 9; i++) copy.field[i / 3][i % 3] = field[i / 3][i % 3];
				player2Fields.push_back(copy);
				player2Moves.push_back(maxValueIndex);
			}


			//Если мы тут, то выбрали свободную клеточку
			//значит, мы можем ходить уже

			field[maxValueIndex / 3][maxValueIndex % 3] = nowGoes;
			
			if (nowGoes == 1) nowGoes = 2;
			else nowGoes = 1;

		}

		if (g % 500 == 0) cout << "Game #" << g << "\tWrong choices of AI: " << wrongChoice << "\tWinner: " << checkWinner() << endl;

		//После окончания партии узнаем имя победителя:
		int winner = checkWinner();
		
		//Если это не ничья, то исправляем веса - тренируем ИИ:
		if ((wrongChoice == 0)) {

			//Скорость обучения
			double learningRate = 0.01;

			if ((winner == 1) or (winner == 0)){
				//Если победил игрок №1, то
				//у игрока 2 исправляем его решение на 0, а остальные делаем одинаковыми

				for (int i = 0; i < player2Moves.size(); i++) {

					//Прогоняем нейросеть через нужную карту 

					double input[9];

					for (int j = 0; j < 9; j++) {
						if (player2Fields[i].field[j / 3][j % 3] == 0) input[i] = 0.0;
						if (player2Fields[i].field[j / 3][j % 3] == 1) input[i] = -1.0;
						if (player2Fields[i].field[j / 3][j % 3] == 2) input[i] = 1.0;
					}

					player.ForwardFeed(input);

					double answers[9];
					
					//Считаем количество пустых клеточек

					int emptyCellCounter = 0;

					for (int j = 0; j < 9; j++) {
						if (player2Fields[i].field[j / 3][j % 3] == 0) emptyCellCounter++;
					}

					//Вычисляем правильные ответы
					for (int j = 0; j < 9; j++) {
						if ((j == player2Moves[i]) or (player2Fields[i].field[j / 3][j % 3] != 0)) answers[j] = 0.0;
						else answers[j] = 1.0 /	double(emptyCellCounter - 1);
					}

					player.BackPropogation(answers, learningRate);
				}
			}
			if ((winner == 2) or (winner == 0)) {
				//Если победил игрок №2, то
				//у игрока 1 исправляем его решение на 0, а остальные делаем одинаковыми

				for (int i = 0; i < player1Moves.size(); i++) {

					//Прогоняем нейросеть через нужную карту 

					double input[9];

					for (int j = 0; j < 9; j++) {
						if (player1Fields[i].field[j / 3][j % 3] == 0) input[i] = 0.0;
						if (player1Fields[i].field[j / 3][j % 3] == 1) input[i] = 1.0;
						if (player1Fields[i].field[j / 3][j % 3] == 2) input[i] = -1.0;
					}

					player.ForwardFeed(input);

					double answers[9];

					//Считаем количество пустых клеточек

					int emptyCellCounter = 0;

					for (int j = 0; j < 9; j++) {
						if (player1Fields[i].field[j / 3][j % 3] == 0) emptyCellCounter++;
					}

					//Вычисляем правильные ответы
					for (int j = 0; j < 9; j++) {
						if ((j == player1Moves[i]) or (player1Fields[i].field[j / 3][j % 3] != 0)) answers[j] = 0.0;
						else answers[j] = 1.0 / double(emptyCellCounter - 1);
					}

					player.BackPropogation(answers, learningRate);
				}
			}
		}

	}

	player.SaveWeights("weights_tic_tac_toe.txt");

	return 0;
}