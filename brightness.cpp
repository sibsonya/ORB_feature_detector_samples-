#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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

		// calculation of max and min idstance between keypoints
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		// remember good matches
		vector<DMatch>good_matches;
		map<pair<float, float>, pair<int, Mat>>::iterator it;
		for (int i = 0; i < descriptors_1.rows; i++) {
			if (matches[i].distance <= max(2.8 * min_dist, 0.05)) {
				good_matches.push_back(matches[i]);
				Mat d = descriptors_1.row(i);
				it = points.find({ keypoints_1[i].pt.x, keypoints_1[i].pt.y });
				if (it != points.end()) {
					(it->second).first += 1;
				}
				else {
					points.insert({ { keypoints_1[i].pt.x, keypoints_1[i].pt.y }, { 1, d } });
				}
			}
		}
		drawKeypoints(img_2, keypoints_2, keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}
	//namedWindow("match result", WINDOW_NORMAL);
	imwrite("brightness.jpg", img_matches);
	imshow("match result", img_matches);
	waitKey();
	return 0;
}

int main() {
	Mat img = imread("testimg.jpg");
	map <pair<float, float>, pair<int, Mat>> points;
	Mat input;
	cvtColor(img, input, 0);
	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	Ptr<ORB> orb = ORB::create();
	orb->detect(input, keypoints_1);
	orb->compute(input, keypoints_1, descriptors_1);

	//changing brightness
	std::vector<int> bright_coeff = { -80, -50, -10, 10, 50, 80 };
	for (int i = 0; i != 6; ++i) {
		Mat new_image = Mat::zeros(img.size(), img.type());
		for (int y = 0; y < img.rows; y++) {
			for (int x = 0; x < img.cols; x++) {
				for (int c = 0; c < img.channels(); c++) {
					new_image.at<Vec3b>(y, x)[c] =
						saturate_cast<uchar>(img.at<Vec3b>(y, x)[c] + bright_coeff[i]);
				}
			}
		}
		matching(input, new_image, keypoints_1, descriptors_1, points);
	}
	return 0;
}