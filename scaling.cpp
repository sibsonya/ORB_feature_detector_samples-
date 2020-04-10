#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

std::vector<std::pair<cv::KeyPoint, cv::Mat>> best_keypoints;
cv::Mat img;

int matching(Mat img_1, Mat img_2, vector<KeyPoint> keypoints_1, Mat descriptors_1, map<pair<float, float>, pair<int, Mat>> &points) {
	vector<KeyPoint> keypoints_2;
	Mat descriptors_2;
	Ptr<ORB> orb = ORB::create();
	orb->detect(img_2, keypoints_2);
	orb->compute(img_2, keypoints_2, descriptors_2);

	BFMatcher matcher;
	vector<DMatch> matches;
	Mat img_matches, keypoints;
	if (!descriptors_1.empty() && !descriptors_2.empty()) {
		matcher.match(descriptors_1, descriptors_2, matches);
		double max_dist = 0; double min_dist = 100;

		// calculation of max and min distance between keypoints
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		// Remember good matches
		std::vector<cv::DMatch>good_matches;
		map<pair<float, float>, pair<int, Mat>>::iterator it;
		for (int i = 0; i < descriptors_1.rows; i++) {
			if (matches[i].distance <= max(2.8 * min_dist, 0.05)) {
				good_matches.push_back(matches[i]);
				Mat d = descriptors_1.row(i);
				it = points.find({ keypoints_1[i].pt.x, keypoints_1[i].pt.y });
				if (it != points.end()) {
					(it->second).first += 1;
				} else {
					points.insert({ { keypoints_1[i].pt.x, keypoints_1[i].pt.y }, { 1, d } });
				}
			}
		}
		drawKeypoints(img_2, keypoints_2, keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}
	namedWindow("match result", WINDOW_NORMAL);
	imshow("match result", img_matches);
	cv::waitKey();
	return 0;
}

int main() {
	img = cv::imread("testimg.jpg");
	map <std::pair<float, float>, std::pair<int, cv::Mat>> points;
	Mat input;
	cvtColor(img, input, 0);
	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	Ptr<ORB> orb = ORB::create();
	orb->detect(input, keypoints_1);
	orb->compute(input, keypoints_1, descriptors_1);

	// making scaled images and matching
	std::vector<float> coeff = { 5.0f, 2.0f, 1.2f,  0.8f, 0.6f, 0.5f };
	for (int i = 0; i != 6; ++i) {
		cv::Mat scaled_img;
		cv::resize(img, scaled_img, cv::Size(0, 0), coeff[i], coeff[i], cv::INTER_LANCZOS4);
		matching(input, scaled_img, keypoints_1, descriptors_1, points);
	}
	return 0;
}