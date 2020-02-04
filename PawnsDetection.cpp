#include "PawnsDetection.h"

using namespace cv;
using namespace std;


//take the black & white image in input
array<array<bool, 8>, 8> findPawns(const Mat& I) {
	//compute the score of each case to determine if there is a pawn
	array<array<array<int, 5>, 8>, 8> whitePixels = { 0 };
	array<array<bool, 8>, 8> boardPawns = { false };
	int m = I.rows, n = I.cols;
	int km = m / 8, kn = n / 8;
	for (int ii = 0; ii < m; ii++) {
		for (int jj = 0; jj < n; jj++) {
			int x = (int)ii % km;
			int y = (int)jj % kn;
			if ((I.at<uchar>(ii, jj) > 230)& (x > 3)& (x < 96)& (y > 3)& (y < 96)& (!((x > 40)& (x < 60)& (y > 40)& (y < 60)))) { //find white pixels not too close from the border (ligns of the board) & not to close of the center of each case
				whitePixels[(int)ii / km][(int)jj / kn][4] += 1;
				if (((int)ii % km > 50)& ((int)jj % kn > 50)) {
					whitePixels[(int)ii / km][(int)jj / kn][0] += 1;
				}
				else if (((int)ii % km > 50)& ((int)jj % kn <= 50)) {
					whitePixels[(int)ii / km][(int)jj / kn][1] += 1;
				}
				else if (((int)ii % km <= 50) & ((int)jj % kn > 50)) {
					whitePixels[(int)ii / km][(int)jj / kn][2] += 1;
				}
				else {
					whitePixels[(int)ii / km][(int)jj / kn][3] += 1;
				}
			}
		}
	}
	//add detected pawns
	int odd_even_count[2] = { 0 };
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 8; jj++) {
			int detected_quarter = 0;
			if (whitePixels[ii][jj][4] > 30) {
				for (int aa = 0; aa < 4; aa++) {
					if (whitePixels[ii][jj][aa] > 4) {
						detected_quarter += 1;
					}
				}
				if ((detected_quarter >= 3) | (whitePixels[ii][jj][4] >= 50)) {
					boardPawns[ii][jj] = true;
					odd_even_count[(ii + jj) % 2] += 1;
				}
			}
		}
	}
	//only keep the pawns on the right cases (even or odd)
	int odd_even_sum = 0;
	if (odd_even_count[1] > odd_even_count[0]) {
		odd_even_sum = 1;
	}
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 8; jj++) {
			if ((ii + jj) % 2 != odd_even_sum) {
				boardPawns[ii][jj] = false;
			}
		}
	}
	return boardPawns;
}

//take the gray-scaled image and the matrix giving the presence of the pawns on the board in input
array<array<int, 8>, 8> findColorPawns(const Mat& I, array<array<bool, 8>, 8> detectedPawns) {
	array<array<int, 8>, 8> colorPawns = { 0 };
	array<array<Vec3f, 8>, 8> RGBcolorPawns = { 0 };
	//computing the mean on each case of the grid
	int m = I.rows, n = I.cols;
	int km = m / 8, kn = n / 8;
	for (int ii = 0; ii < m; ii++) {
		for (int jj = 0; jj < n; jj++) {
			if (detectedPawns[(int)ii / km][(int)jj / kn]) {
				RGBcolorPawns[(int)ii / km][(int)jj / kn] += I.at<Vec3b>(ii, jj);
			}
		}
	}
	// cluster the colors in two groups
	int n_it = 5;
	vector<Vec3b> average_colors;
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 8; jj++) {
			RGBcolorPawns[ii][jj] /= kn * km;
			if (detectedPawns[ii][jj]) average_colors.push_back(RGBcolorPawns[ii][jj]);
		}
	}
	Vec3f average_col1 = Vec3f(0, 0, 0), average_col2 = Vec3f(200, 200, 200);
	Vec3f next_average_col1 = Vec3f(0, 0, 0), next_average_col2 = Vec3f(200, 200, 200);
	int count_col1 = 0, count_col2 = 1;
	for (int ii = 0; ii < n_it; ii++) {
		next_average_col1 = Vec3f(0, 0, 0), next_average_col2 = Vec3f(200, 200, 200);
		count_col1 = 0, count_col2 = 1;
		for (Vec3f color : average_colors) {
			if (norm(color - average_col1) < norm(color - average_col2)) {
				count_col1++;
				next_average_col1 += color;
			}
			else {
				count_col2++;
				next_average_col2 += color;
			}
		}
		next_average_col1 /= count_col1;
		next_average_col2 /= count_col2;
		average_col1 = next_average_col1;
		average_col2 = next_average_col2;
	}
	// get the final group for each case
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 8; jj++) {
			Vec3f color = RGBcolorPawns[ii][jj];
			if (detectedPawns[ii][jj]) {
				if (norm(color - average_col1) < norm(color - average_col2)) {
					colorPawns[ii][jj] = 1;
				}
				else {
					colorPawns[ii][jj] = -1;
				}
			}
		}
	}
	return colorPawns;
}

vector<Vec3f> get_circles(const Mat& I) {
	vector<Vec3f> circles;
	float sensibility = 15;
	HoughCircles(I, circles, HOUGH_GRADIENT, 1, I.rows / 8, 100, sensibility, I.rows / 32, I.rows / 16);
	while (circles.size() > 25) {
		circles.clear();
		sensibility += 2;
		HoughCircles(I, circles, HOUGH_GRADIENT, 1, I.rows / 8, 100, sensibility, I.rows / 32, I.rows / 16);
	}
	return circles;
}

void draw_circles(vector<Vec3f> circles, Mat I) {
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(I, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(I, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	imshow("circles", I);
}

array<array<bool, 8>, 8> findPawns(vector<Vec3f> circles, const Mat& I) {
	array<array<bool, 8>, 8> boardPawns = { false };
	int m = I.rows, n = I.cols;
	int km = m / 8, kn = n / 8;
	for (Vec3f circle : circles) {
		Point2f center(circle[0], circle[1]);
		float radius = circle[2];
		int i = floor(center.y / km);
		int j = floor(center.x / kn);
		if (floor((center.y + radius / 3) / km) == i
			&& floor((center.y - radius / 3) / km) == i
			&& floor((center.x - radius / 3) / kn) == j
			&& floor((center.x - radius / 3) / kn) == j
			) boardPawns[i][j] = true;
	}
	return boardPawns;
}

void print_pawns(array<array<int, 8>, 8 > pawns) {
	cout << ">>> printing pawns..." << endl;
	for (auto line : pawns) {
		for (auto col : line) {
			cout << col << ", ";
		}
		cout << endl;
	}
	cout << endl;
}