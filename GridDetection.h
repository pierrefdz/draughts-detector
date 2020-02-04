#pragma once

#ifndef GRID_DETECTION_H_INCLUDED
#define GRID_DETECTION_H_INCLUDED

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <map> 
#include <numeric>  

void add_line(cv::Vec2f l, cv::Mat cdst);
bool compare_lines_r(cv::Vec2f l1, cv::Vec2f l2);
bool compare_lines_theta(cv::Vec2f l1, cv::Vec2f l2);
std::vector<std::vector<cv::Vec2f>> theta_hist(std::vector<cv::Vec2f> lines, int res);
float sum_theta(float sum, cv::Vec2f l1);
cv::Vec2f sum_line(cv::Vec2f sum, cv::Vec2f l1);
std::vector<std::vector<cv::Vec2f>> main_directions(std::vector<cv::Vec2f> lines);
std::array<std::vector<cv::Vec2f>, 2> cluster_directions(std::vector<std::vector<cv::Vec2f>> lines_by_dir);
void print_lines(std::vector<cv::Vec2f> lines);
void print_hist(std::vector<std::vector<cv::Vec2f>> hist);
std::vector<cv::Vec2f> get_lines(const cv::Mat& dst);
void draw_lines(std::vector<cv::Vec2f> lines, const cv::Mat& cdst);
void draw_batches(std::vector<std::vector<cv::Vec2f>> batches, const cv::Mat& cdst);
std::vector<std::vector<cv::Vec2f>> r_hist(std::vector<cv::Vec2f> lines, int res, float max_r);
std::vector<std::vector<cv::Vec2f>> main_rs(std::vector<cv::Vec2f> lines, float max_r);
std::vector<cv::Vec2f> average_lines(std::vector<std::vector<cv::Vec2f>> hist);

cv::Point2f intersection(cv::Vec2f l1, cv::Vec2f l2);
float scoreBoard(const cv::Mat& I);
float surfaceQuad(std::vector<cv::Point2f> corners);
bool possibleBoard(const cv::Mat& I, float score_max);
std::vector<cv::Point2f> order(std::vector<cv::Point2f> corners);
std::vector <cv::Point2f> find_best_corners(const cv::Mat& I, std::array<std::vector<cv::Vec2f>, 2> clusters);

#endif