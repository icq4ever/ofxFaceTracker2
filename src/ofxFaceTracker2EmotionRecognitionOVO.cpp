#include "ofxFaceTracker2EmotionRecognitionOVO.h"

ofxFaceTracker2EmotionRecognitionOVO::ofxFaceTracker2EmotionRecognitionOVO(){
	std::vector<sample_type> samples;

	dlib::deserialize(ofFile(shapeFileName).path()) >> sp;
	dlib::deserialize(ofFile(emotionFileName1).path()) >> ep1;
	dlib::deserialize(ofFile(emotionFileName2).path()) >> ep2;
	dlib::deserialize(ofFile(emotionFileName3).path()) >> ep3;
	dlib::deserialize(ofFile(emotionFileName4).path()) >> ep4;
	dlib::deserialize(ofFile(emotionFileName5).path()) >> ep5;
	dlib::deserialize(ofFile(emotionFileName6).path()) >> ep6;
}

ofxFaceTracker2EmotionRecognitionOVO::~ofxFaceTracker2EmotionRecognitionOVO(){

}

void ofxFaceTracker2EmotionRecognitionOVO::getEmotionFromImage(dlib::cv_image<unsigned char> image){
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	sample_type sample;
//	std::vector<sample_type>samples;

	dlib::array2d<dlib::rgb_pixel> img;

	dlib::assign_image(img, dlib::cv_image<unsigned char>(image));

	std::vector<dlib::rectangle> faceRectangles = detector(img);
	std::vector<dlib::full_object_detection> facialFeatures;
	dlib::full_object_detection feature = sp(img, faceRectangles[0]);

	int l = 0;
	for(int j = 0; j < 68; j++){
		for(int k = 0; k < j; k++,l++){
			sample(l) = length(feature.part(j),feature.part(k));
			l++;
			sample(l) = slope(feature.part(j),feature.part(k));
		}
	}
//	samples.push_back(sample);

//	for(int i=0; i<samples.size(); i++){

		std::vector<double> prob;
		prob = svmMulticlass(sample);
		prob = probablityCalculator(prob);

		// get maximum probabiliry index from vector "prob"
		std::vector<double>::iterator result;
		result = std::max_element(prob.begin(), prob.end());
		int idx = std::distance(prob.begin(), result);


		switch(idx){
		case 0:
			std::cout <<"face is nutural" << std::endl;
			break;
		case 1 :
			std::cout << "face is happy" << std::endl;
			break;
		case 2:
			std::cout << "face is sad" << std::endl;
			break;
		case 3:
			std::cout << "face is surprise" << std::endl;
			break;
		default:
			std::cout << "unknown" << std::endl;
			break;
		}
//	}
}

std::vector<double> ofxFaceTracker2EmotionRecognitionOVO::svmMulticlass(sample_type sample)
{
	std::vector<double> probs;
	probs.push_back(ep1(sample));
	probs.push_back(ep2(sample));
	probs.push_back(ep3(sample));
	probs.push_back(ep4(sample));
	probs.push_back(ep5(sample));
	probs.push_back(ep6(sample));

	return probs;
}

void ofxFaceTracker2EmotionRecognitionOVO::getEmotion(dlib::full_object_detection shape){
//void ofxFaceTracker2EmotionRecognitionOVO::getEmotion(dlib::full_object_detection shape, dlib::rectangle rect){
//	samples = getAllAttributes(shape);


//	std::vector<sample_type> samples;
	sample_type sample;

	int l = 0;
	for(int j = 0; j < 68; j++){
		for(int k = 0; k < j; k++,l++){
//			sample(l) = length(shape.part(j),shape.part(k));
//			l++;
//			sample(l) = slope(shape.part(j),shape.part(k));
			sample(l) = length(shape.part(j),shape.part(k));
			l++;
			sample(l) = slope(shape.part(j), shape.part(k));
		}
	}
//	samples.push_back(sample);
	std::cout << "l : " << l << std::endl;
	std::vector<double> prob;
	prob = svmMulticlass(sample);
	prob = probablityCalculator(prob);

	// get maximum probabiliry index from vector "prob"
	std::vector<double>::iterator result;
	result = std::max_element(prob.begin(), prob.end());
	int idx = result - prob.begin();
//	int idx = std::distance(prob.begin(), result);


	switch(idx){
	case 0:
		std::cout <<"face is nutural" << std::endl;
		break;
	case 1 :
		std::cout << "face is happy" << std::endl;
		break;
	case 2:
		std::cout << "face is sad" << std::endl;
		break;
	case 3:
		std::cout << "face is surprise" << std::endl;
		break;
	default:
		std::cout << "unknown" << std::endl;
		break;
	}

//	std::max_element(prob, prob + 4) - prob;
	/*
	std::cout << "probablity that face is Neutral  :" << prob[0] << std::endl;
	std::cout << "probablity that face is Happy    :" << prob[1] << std::endl;
	std::cout << "probablity that face is Sad      :" << prob[2] << std::endl;
	std::cout << "probablity that face is Surprise :" << prob[3] << "\n\n\n"*/;
}

double ofxFaceTracker2EmotionRecognitionOVO::length(dlib::point a, dlib::point b){
	int x1,y1,x2,y2;
	double dist;
	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();

	dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
	dist = sqrt(dist);
	return dist;
}

double ofxFaceTracker2EmotionRecognitionOVO::slope (dlib::point a, dlib::point b){
	int x1,y1,x2,y2;

	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();
	if((x1-x2) == 0)
		if((y1-y2) > 0)
			return (M_PI/2);
		else
			return (-M_PI/2);
	else
		return atan(double(y1-y2))/(x1-x2);
}

std::vector<double> ofxFaceTracker2EmotionRecognitionOVO::probablityCalculator(std::vector<double> P){
	std::vector<double> EmoProb(4);
	float e[4],temp;

	for(int i=0;i < 6;i++){

		P.push_back(1-P[i]);
	}

	e[0] = P[0]+P[1]+P[2];
	e[1] = P[3]+P[4]+P[6];
	e[2] = P[5]+P[7]+P[9];
	e[3] = P[8]+P[10]+P[11];

	int  i, j;
	int t[]={0,1,2,3};

	for(i=0;i<4;i++){
		for(j=i+1;j<4;j++){
			if(e[i]<e[j]){
				temp=e[i];
				e[i]=e[j];
				e[j]=temp;

				temp=t[i];
				t[i]=t[j];
				t[j]=temp;
			}
		}
	}

	e[0] = e[0]/3;
	e[1]= (1-e[0])*e[1]/3;
	e[2]=(1-e[0]-e[1])*e[2]/3;
	e[3]=(1-e[0]-e[1]-e[2]);

	for(i=0;i<4;i++){
		EmoProb[t[i]]=e[i];
	}

	return EmoProb;
}
