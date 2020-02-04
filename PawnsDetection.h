#pragma once

#ifndef PAWNSDETECTION_H_INCLUDED
#define PAWNSDETECTION_H_INCLUDED

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

std::array<std::array<bool, 8>, 8> findPawns(const cv::Mat& I);
std::array<std::array<int, 8>, 8> findColorPawns(const cv::Mat& I, std::array<std::array<bool, 8>, 8> detectedPawns);

std::vector<cv::Vec3f> get_circles(const cv::Mat& I);
void draw_circles(std::vector<cv::Vec3f> circles, cv::Mat I);
std::array<std::array<bool, 8>, 8> findPawns(std::vector<cv::Vec3f> circles, const cv::Mat& I);
void print_pawns(std::array<std::array<int, 8>, 8> pawns);


#endif