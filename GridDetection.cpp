#include "GridDetection.h"

using namespace cv;
using namespace std;

void add_line(Vec2f l, Mat cdst) {
	float rho = l[0], theta = l[1];
	Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1000 * (-b));
	pt2.y = cvRound(y0 - 1000 * (a));
	line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
}

bool compare_lines_r(Vec2f l1, Vec2f l2) {
	return (abs(l1[0]) < abs(l2[0]));
}

bool compare_lines_theta(Vec2f l1, Vec2f l2) {
	return (l1[1] < l2[1]);
}

vector<vector<Vec2f>> theta_hist(vector<Vec2f> lines, int res) {
	vector<vector<Vec2f>> hist = vector<vector<Vec2f>>(res);
	float step = CV_PI / (float)res;
	for (size_t ii = 0; ii < lines.size(); ii++)
	{
		float theta = lines[ii][1];
		int index = floor(theta / step);
		hist[index].push_back(lines[ii]);
	}
	return hist;
}

float sum_theta(float sum, Vec2f l1) {
	return sum + l1[1];
}

Vec2f sum_line(Vec2f sum, Vec2f l1) {
	return sum + l1;
}

vector<vector<Vec2f>> main_directions(vector<Vec2f> lines) {
	int res = 16;
	vector<vector<Vec2f>> hist = theta_hist(lines, res);
	vector<vector<Vec2f>> main_directions;
	for (auto batch : hist) {
		if (batch.size() > 0.7 * lines.size() / res) {
			main_directions.push_back(batch);
		}
	}
	return main_directions;
}

array<vector<Vec2f>, 2> cluster_directions(vector<vector<Vec2f>> lines_by_dir) {
	int n_it = 5;
	vector<float> average_thetas;
	for (vector<Vec2f> batch : lines_by_dir) {
		average_thetas.push_back(accumulate(batch.begin(), batch.end(), 0.0, sum_theta) / batch.size());
	}
	float average_dir1 = 0, average_dir2 = CV_PI / 2;
	float next_average_dir1 = 0, next_average_dir2 = 0;
	int count_dir1 = 0, count_dir2 = 0;
	for (int ii = 0; ii < n_it; ii++) {
		next_average_dir1 = 0; next_average_dir2 = 0;
		count_dir1 = 0; count_dir2 = 0;
		for (float theta : average_thetas) {
			if (theta > CV_PI / 2) theta = CV_PI - theta;
			if (abs(theta - average_dir1) < abs(theta - average_dir2)) {
				count_dir1++;
				next_average_dir1 += theta;
			}
			else {
				count_dir2++;
				next_average_dir2 += theta;
			}
		}
		next_average_dir1 /= count_dir1;
		next_average_dir2 /= count_dir2;
		average_dir1 = next_average_dir1;
		average_dir2 = next_average_dir2;
	}
	array<vector<Vec2f>, 2> clusters;
	for (int ii = 0; ii < average_thetas.size(); ii++) {
		float theta = (average_thetas[ii] > CV_PI / 2) ? CV_PI - average_thetas[ii] : average_thetas[ii];
		if (abs(theta - average_dir1) < abs(theta - average_dir2)) {
			clusters[0].insert(clusters[0].end(), lines_by_dir[ii].begin(), lines_by_dir[ii].end());
		}
		else {
			clusters[1].insert(clusters[1].end(), lines_by_dir[ii].begin(), lines_by_dir[ii].end());
		}
	}
	return clusters;
}

void print_lines(vector<Vec2f> lines) {
	cout << ">>> printing lines ..." << endl;
	for (size_t ii = 0; ii < lines.size(); ii++) {
		std::cout << lines[ii] << std::endl;
	};
	cout << endl;
}

void print_hist(vector<vector<Vec2f>> hist) {
	for (size_t ii = 0; ii < hist.size(); ii++) {
		cout << " ----------------------- " << endl;
		for (size_t jj = 0; jj < hist[ii].size(); jj++) {
			std::cout << hist[ii][jj] << std::endl;
		};
	};
}

vector<Vec2f> get_lines(const Mat& dst) {
	vector<Vec2f> lines;
	int minLength = 5; //min length for segments to be detected
	int maxGapSegments = 1; //max lenght to join segments
	int thres = 200; //number of intersections in the houghline curve
	while (lines.size() < 120) {
		lines.clear();
		HoughLines(dst, lines, 1, CV_PI / 360, thres);
		thres -= 10;
	}
	HoughLines(dst, lines, 1, CV_PI / 360, thres);
	return lines;
}

void draw_lines(vector<Vec2f> lines, const Mat& cdst) {
	for (size_t ii = 0; ii < lines.size(); ii++) {
		add_line(lines[ii], cdst);
	};
	imshow("detected lines", cdst);
}

void draw_batches(vector<vector<Vec2f>> batches, const Mat& cdst) {
	for (auto batch : batches) {
		for (auto line : batch) {
			add_line(line, cdst);
		};
	};
	imshow("detected lines", cdst);
}

vector<vector<Vec2f>> r_hist(vector<Vec2f> lines, int res, float max_r) {
	vector<vector<Vec2f>> hist = vector<vector<Vec2f>>(res);
	float step = 2 * max_r / (float)res;
	for (size_t ii = 0; ii < lines.size(); ii++)
	{
		float r = lines[ii][0];
		int index = floor((max_r + r) / step);
		hist[index].push_back(lines[ii]);
	}
	return hist;
}

vector<vector<Vec2f>> main_rs(vector<Vec2f> lines, float max_r) {
	int res = 400;
	vector<vector<Vec2f>> hist = r_hist(lines, res, max_r);
	vector<vector<Vec2f>> main_rs;
	for (auto batch : hist) {
		if (batch.size() > 0.7 * lines.size() / res) {
			main_rs.push_back(batch);
		}
	}
	return main_rs;
}

vector<Vec2f> average_lines(vector<vector<Vec2f>> hist) {
	vector<Vec2f> average_lines;
	for (auto batch : hist) {
		Vec2f acc_line = accumulate(batch.begin(), batch.end(), Vec2f(0, 0), sum_line);
		acc_line[0] /= batch.size();
		acc_line[1] /= batch.size();
		average_lines.push_back(acc_line);
	}
	return average_lines;
}

Point2f intersection(Vec2f l1, Vec2f l2) {
	float r1 = l1[0];
	float r2 = l2[0];
	float theta1 = l1[1];
	float theta2 = l2[1];
	Matx22f A(cos(theta1), sin(theta1),
		cos(theta2), sin(theta2));
	Matx21f b(r1, r2);
	Matx21f x = A.inv() * b;
	return Point2f(x(0), x(1));
}

float scoreBoard(const Mat& I) {
	//compute the mean value of each case
	float score = 0.0;
	float colorsBoard[8][8];
	int m = I.rows, n = I.cols;
	int km = m / 8, kn = n / 8;
	for (int ii = 0; ii < m; ii++) {
		for (int jj = 0; jj < n; jj++) {
			colorsBoard[ii / km][jj / kn] += I.at<uchar>(ii, jj);
		}
	}
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 8; jj++) {
			colorsBoard[ii][jj] /= kn * km;
		}
	}
	//compute the score of the ligns
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 7; jj++) {
			score += pow(-1, ii + jj) * (colorsBoard[ii][jj] - colorsBoard[ii][jj + 1]); //sum on the lines
			score += pow(-1, ii + jj) * (colorsBoard[jj][ii] - colorsBoard[jj + 1][ii]); //sum on the columns
		}
	}
	return abs(score);
}

float surfaceQuad(vector<Point2f> corners) {
	Vec2f c0(corners[0].x, corners[0].y);
	Vec2f c1(corners[1].x, corners[1].y);
	Vec2f c2(corners[2].x, corners[2].y);
	Vec2f c3(corners[3].x, corners[3].y);
	float p = norm(c2 - c0);
	float q = norm(c3 - c1);
	float a = norm(c1 - c0);
	float b = norm(c2 - c1);
	float c = norm(c3 - c2);
	float d = norm(c0 - c3);
	return sqrt(4 * p * p * q * q - (b * b + d * d - a * a - c * c) * (b * b + d * d - a * a - c * c)) / 4;
}

bool possibleBoard(const Mat& I, float score_max) {
	float score_min = 0.9 * score_max;
	//compute the mean value of each case
	float score = 0.;
	float score_borders = 0.;
	float colorsBoard[8][8];
	int m = I.rows, n = I.cols;
	int km = m / 8, kn = n / 8;
	for (int ii = 0; ii < m; ii++) {
		for (int jj = 0; jj < n; jj++) {
			colorsBoard[ii / km][jj / kn] += I.at<uchar>(ii, jj);
		}
	}
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 8; jj++) {
			colorsBoard[ii][jj] /= kn * km;
		}
	}
	//compute the score of the lines
	for (int ii = 0; ii < 8; ii++) {
		for (int jj = 0; jj < 7; jj++) {
			score += pow(-1, ii + jj) * (colorsBoard[ii][jj] - colorsBoard[ii][jj + 1]); //sum on the lines
			score += pow(-1, ii + jj) * (colorsBoard[jj][ii] - colorsBoard[jj + 1][ii]); //sum on the columns
			if (ii == 0 || ii == 7) {
				score_borders += pow(-1, ii + jj) * (colorsBoard[ii][jj] - colorsBoard[ii][jj + 1]); //sum on the lines
				score_borders += pow(-1, ii + jj) * (colorsBoard[jj][ii] - colorsBoard[jj + 1][ii]); //sum on the columns
			}
		}
	}
	if ((abs(score) > score_min) && (abs(score_borders) > 0.22 * abs(score))) {
		return true;
	}
	else {
		return false;
	}
}

vector<Point2f> order(vector<Point2f> corners) {
	Point2f middle;
	for (auto p : corners) middle += p;
	middle /= 4.0;
	vector<float> angles;
	Vec3f vM0(corners[0].x - middle.x, corners[0].y - middle.y, 0);
	vM0 = normalize(vM0);
	for (int ii = 1; ii < 4; ii++) {
		Vec3f vM(corners[ii].x - middle.x, corners[ii].y - middle.y, 0);
		vM = normalize(vM);
		float angle = acos(vM0[0] * vM[0] + vM0[1] * vM[1]);
		if (vM0.cross(vM)[2] < 0) {
			angles.insert(angles.begin(), angle);
			Point2f c = corners[ii];
			corners.erase(corners.begin() + ii);
			auto it = corners.begin();
			corners.insert(it + 1, c);
		}
		else {
			angles.push_back(angle);
		}
	}
	if ((angles[0] > angles[1]) && (angles[0] > angles[2])) {
		swap(corners[1], corners[2]);
	}
	else if ((angles[2] > angles[0]) && (angles[2] > angles[1])) {
		swap(corners[2], corners[3]);
	}
	return corners;
}

vector<Point2f> find_best_corners(const Mat& I, array<vector<Vec2f>, 2> clusters) {
	int n0 = clusters[0].size();
	int n1 = clusters[1].size();
	float score;
	float max = 0;
	vector<Point2f> corners;
	vector<Point2f> squareCorners = { Point2f(0,0),Point2f(0,128),
									Point2f(128,128), Point2f(128,0) };
	Mat H;
	vector<Point2f> corners_to_return;
	for (int ii = 0; ii < 100; ii++) {
		int r0 = rand() % n0 / 4, r1 = n0 - 1 - rand() % n0 / 4,
			r2 = rand() % n1 / 4, r3 = n1 - 1 - rand() % n1 / 4;
		//cout << r0 << " " << r1 << " " << r2 << " " << r3 << endl;
		Vec2f l0 = clusters[0][r0], l1 = clusters[0][r1],
			l2 = clusters[1][r2], l3 = clusters[1][r3];
		corners = { intersection(l0,l2),intersection(l0,l3),
					intersection(l3,l1), intersection(l2,l1) };
		H = findHomography(corners, squareCorners);
		Mat img_warp;
		warpPerspective(I, img_warp, H, Size(128, 128));
		score = scoreBoard(img_warp);
		if (score > max) {
			max = score;
			corners_to_return = corners;
		}
	}
	float size;
	float size_max = 0;
	for (int ii = 0; ii < 100; ii++) {
		int r0 = rand() % n0 / 4, r1 = n0 - 1 - rand() % n0 / 4,
			r2 = rand() % n1 / 4, r3 = n1 - 1 - rand() % n1 / 4;
		//cout << r0 << " " << r1 << " " << r2 << " " << r3 << endl;
		Vec2f l0 = clusters[0][r0], l1 = clusters[0][r1],
			l2 = clusters[1][r2], l3 = clusters[1][r3];
		corners = { intersection(l0,l2),intersection(l0,l3),
					intersection(l3,l1), intersection(l2,l1) };
		H = findHomography(corners, squareCorners);
		Mat img_warp;
		warpPerspective(I, img_warp, H, Size(128, 128));
		//score = scoreBoard(img_warp);
		if (possibleBoard(img_warp, max)) {
			size = surfaceQuad(corners);
			if (size > size_max) {
				size_max = size;
				corners_to_return = corners;
			}
		}
	}
	return corners_to_return;
}

