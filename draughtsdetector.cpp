#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <map> 
#include <numeric>  

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
	return (l1[0] < l2[0]);
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
	for (auto batch : lines_by_dir) {
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
	int thres = 80; //number of intersections in the houghline curve
	int minLength = 5; //min length for segments to be detected
	int maxGapSegments = 0.3 * dst.rows; //max lenght to join segments
	HoughLines(dst, lines, 1, CV_PI / 360, thres, minLength, maxGapSegments);
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
	int res = 100;
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

array<Point2f, 4> quadrant(array<vector<Vec2f>, 2> batches) {

}

int main()
{
	Mat src;
	src = imread("../data/draught1.jpg");

	resize(src, src, Size(800, 800));
	imshow("source", src);

	Mat src_gray, dst, cdst;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	GaussianBlur(src_gray, src_gray, Size(11, 11), 0);
	Canny(src_gray, dst, 50, 100, 3);
	cvtColor(dst, cdst, 0);

	vector<Vec2f> lines = get_lines(dst);
	float max_r = sqrt(src.rows * src.rows + src.cols * src.cols);

	vector<Vec2f> real_lines;
	for (auto line : lines) {
		if (abs(line[0]) < max_r) real_lines.push_back(line);
	}
	lines = real_lines;

	cout << ">>> computing main direction batches..." << endl;
	//draw_lines(lines, cdst);
	vector<vector<Vec2f>> lines_by_dir = main_directions(lines);
	array<vector<Vec2f>, 2> clusters_dir = cluster_directions(lines_by_dir);
	//draw_lines(clusters_dir[0], cdst);

	cout << ">>> computing main rs batches..." << endl;
	for (int ii = 0; ii < 2; ii++) {
		vector<vector<Vec2f>> lines_by_r = main_rs(clusters_dir[ii], max_r);
		clusters_dir[ii] = average_lines(lines_by_r);
	}

	draw_lines(clusters_dir[0], cdst);
	draw_lines(clusters_dir[1], cdst);

	//print_hist(lines_by_dir);
	//draw_batches(lines_by_dir, cdst);


	waitKey();
	return 0;
}