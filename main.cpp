#include "main.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

	Mat src;
	src = imread("../data/draught1.jpg");

	resize(src, src, Size(800, 800));
	imshow("source", src);

	Mat src_gray, dst, cdst;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	blur(src_gray, src_gray, Size(4, 4));
	Canny(src_gray, dst, 70, 100, 3);

	cvtColor(dst, cdst, 0);

	cout << ">>> finding the right number of lines..." << endl;
	vector<Vec2f> lines = get_lines(dst);
	float max_r = sqrt(src.rows * src.rows + src.cols * src.cols);

	//draw_lines(lines, cdst);

	vector<Vec2f> real_lines;
	for (auto line : lines) {
		if (abs(line[0]) < max_r) real_lines.push_back(line);
	}
	lines = real_lines;

	cout << ">>> computing main direction batches..." << endl;
	vector<vector<Vec2f>> lines_by_dir = main_directions(lines);

	array<vector<Vec2f>, 2> clusters_dir = cluster_directions(lines_by_dir);

	cout << ">>> computing main rs batches..." << endl;
	vector<vector<Vec2f>> lines_by_r;
	for (int ii = 0; ii < 2; ii++) {
		lines_by_r = main_rs(clusters_dir[ii], max_r);
		clusters_dir[ii] = average_lines(lines_by_r);
	}

	//draw_lines(clusters_dir[0], cdst);
	draw_lines(clusters_dir[1], cdst);

	sort(clusters_dir[0].begin(), clusters_dir[0].end(), compare_lines_r);
	sort(clusters_dir[1].begin(), clusters_dir[1].end(), compare_lines_r);

	cout << ">>> computing possible matches..." << endl;

	vector<Point2f> squareCorners = { Point2f(0,0),Point2f(0,400),
								Point2f(400,400), Point2f(400,0) };
	vector<Point2f> bestCorners = find_best_corners(src, clusters_dir);
	Mat H = findHomography(bestCorners, squareCorners);
	Mat img_warp;
	warpPerspective(src, img_warp, H, Size(400, 400));
	imshow("warp", img_warp);

	cout << ">>> detecting the pawns..." << endl;

	Mat gray_warp;
	cvtColor(img_warp, gray_warp, COLOR_BGR2GRAY);
	blur(gray_warp, gray_warp, Size(2, 2));

	Mat canny_warp;
	Canny(gray_warp, canny_warp, 50, 80, 3);
	imshow("canny_wrap", canny_warp);

	vector<Vec3f> circles = get_circles(canny_warp);
	draw_circles(circles, img_warp);

	//array<array<bool, 8>, 8> bool_pawns = findPawns(canny_warp);
	array<array<bool, 8>, 8> bool_pawns = findPawns(circles, canny_warp);
	array<array<int, 8>, 8> color_pawns = findColorPawns(img_warp, bool_pawns);
	print_pawns(color_pawns);

	waitKey();
	return 0;
};
