#pragma once

#include "ofConstants.h"
#include "ofGraphics.h"
//#include "ofThread.h"
#include "ofxCv.h"


#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>


typedef dlib::matrix<double,4556,1> sample_type;

typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::probabilistic_decision_function<kernel_type> probabilistiopenc_funct_type;
typedef dlib::normalized_function<probabilistiopenc_funct_type> pfunct_type;

class ofxFaceTracker2EmotionRecognitionOVO {
public:
	ofxFaceTracker2EmotionRecognitionOVO();
	~ofxFaceTracker2EmotionRecognitionOVO();


//	void getEmotion(dlib::full_object_detection shape, dlib::rectangle rect);
	void getEmotion(dlib::full_object_detection shape);
	void getEmotionFromImage(dlib::cv_image<unsigned char> image);
private:
	std::string shapeFileName = "shape_predictor_68_face_landmarks.dat";
	std::string emotionFileName1 = "neutral_vs_happy.dat";
	std::string emotionFileName2 = "neutral_vs_sad.dat";
	std::string emotionFileName3 = "neutral_vs_surprise.dat";
	std::string emotionFileName4 = "happy_vs_sad.dat";
	std::string emotionFileName5 = "happy_vs_surprise.dat";
	std::string emotionFileName6 = "sad_vs_surprise.dat";

	double length(dlib::point a, dlib::point b);
	double slope (dlib::point a, dlib::point b);
	std::vector<double> probablityCalculator(std::vector<double> P);

	std::vector<double> svmMulticlass(sample_type sample);

	dlib::shape_predictor sp;
	pfunct_type ep1;
	pfunct_type ep2;
	pfunct_type ep3;
	pfunct_type ep4;
	pfunct_type ep5;
	pfunct_type ep6;
	int faceNumber = 0;
};
