#include "ocr.h"
#include <queue>
#include <numeric>

#define CRNN_LSTM 1

OCR::OCR(const string &model_path) {
    psenet.load_param((model_path+"/psenet_lite_mbv2.param").data());
    psenet.load_model((model_path+"/psenet_lite_mbv2.bin").data());

#if CRNN_LSTM
    crnn_net.load_param((model_path+"/crnn_lite_lstm_v2.param").data());
    crnn_net.load_model((model_path+"/crnn_lite_lstm_v2.bin").data());
    crnn_vertical_net.load_param((model_path+"/crnn_lite_lstm_vertical.param").data());
    crnn_vertical_net.load_model((model_path+"/crnn_lite_lstm_vertical.bin").data());
#else
    crnn_net.load_param((model_path+"/crnn_lite_dw_dense.param").data());
    crnn_net.load_model((model_path+"/crnn_lite_dw_dense.bin").data());
    crnn_vertical_net.load_param((model_path+"/crnn_lite_dw_dense_vertical.param").data());
    crnn_vertical_net.load_model((model_path+"/crnn_lite_dw_dense_vertical.bin").data());
#endif

    angle_net.load_param((model_path+"/shufflenetv2_05_angle.param").data());
    angle_net.load_model((model_path+"/shufflenetv2_05_angle.bin").data());

    //load keys
    ifstream in((model_path+"/keys.txt").data());
	std::string filename;
	std::string line;

	if(in) {
		while(getline(in, line)) {
            alphabetChinese.push_back(line);
		}
	}
	else {
		std::cout <<"no txt file" << std::endl;
	}
}

std::vector<std::string> crnn_deocde(const ncnn::Mat score , std::vector<std::string> alphabetChinese) {
    float *srcdata = (float* ) score.data;
    std::vector<std::string> str_res;
    int last_index = 0;  
    for (int i = 0; i < score.h;i++){
        int max_index = 0;
        
        float max_value = -1000;
        for (int j =0; j< score.w; j++){
            if (srcdata[ i * score.w + j ] > max_value){
                max_value = srcdata[i * score.w + j ];
                max_index = j;
            }
        }
        if (max_index >0 && (not (i>0 && max_index == last_index))  ){
            str_res.push_back(alphabetChinese[max_index-1]);
        }

        last_index = max_index;
    }
    return str_res;
}

cv::Mat resize_img(cv::Mat src,const int long_size) {
    int w = src.cols;
    int h = src.rows;
    float scale = 1.f;
    if (w > h) {
        scale = (float)long_size / w;
        w = long_size;
        h = h * scale;
    }
    else {
        scale = (float)long_size / h;
        h = long_size;
        w = w * scale;
    }
    if (h % 32 != 0) {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0) {
        w = (w / 32 + 1) * 32;
    }
    cv::Mat result;
    cv::resize(src, result, cv::Size(w, h));
    return result;
}

void pse_deocde(ncnn::Mat& features,
                std::map<int, std::vector<cv::Point>>& contours_map,
                const float thresh,
                const float min_area,
                const float ratio
                ) {
    // kernel
    float *srcdata = (float *) features.data;
    std::vector<cv::Mat> kernels;

    float _thresh = thresh;
    cv::Mat scores = cv::Mat::zeros(features.h, features.w, CV_32FC1);
    for (int c = features.c - 1; c >= 0; --c){
        cv::Mat kernel(features.h, features.w, CV_8UC1);
        for (int i = 0; i < features.h; i++) {
            for (int j = 0; j < features.w; j++) {

                if (c==features.c - 1) scores.at<float>(i, j) = srcdata[i * features.w + j + features.w*features.h*c] ;

                if (srcdata[i * features.w + j + features.w*features.h*c ] >= _thresh) {
                    kernel.at<uint8_t>(i, j) = 1;
                } else {
                    kernel.at<uint8_t>(i, j) = 0;
                }
            }
        }
        kernels.push_back(kernel);
        _thresh = thresh * ratio;
    }

    // label
    cv::Mat label;
    std::map<int, int> areas;
    std::map<int, float> scores_sum;
    cv::Mat mask(features.h, features.w, CV_32S, cv::Scalar(0));
    cv::connectedComponents(kernels[features.c  - 1], label, 4);

    for (int y = 0; y < label.rows; ++y) {
        for (int x = 0; x < label.cols; ++x) {
            int value = label.at<int32_t>(y, x);
            float score = scores.at<float>(y,x);
            if (value == 0) continue;
            areas[value] += 1;

            scores_sum[value] += score;
        }
    }

    std::queue<cv::Point> queue, next_queue;

    for (int y = 0; y < label.rows; ++y) {
        for (int x = 0; x < label.cols; ++x) {
            int value = label.at<int>(y, x);

            if (value == 0) continue;
            if (areas[value] < min_area) {
                areas.erase(value);
                continue;
            }

            if (scores_sum[value]*1.0 /areas[value] < 0.93) {
                areas.erase(value);
                scores_sum.erase(value);
                continue;
            }
            cv::Point point(x, y);
            queue.push(point);
            mask.at<int32_t>(y, x) = value;
        }
    }

    //textline
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int idx = features.c  - 2; idx >= 0; --idx) {
        while (!queue.empty()) {
            cv::Point point = queue.front(); queue.pop();
            int x = point.x;
            int y = point.y;
            int value = mask.at<int32_t>(y, x);

            bool is_edge = true;
            for (int d = 0; d < 4; ++d) {
                int _x = x + dx[d];
                int _y = y + dy[d];

                if (_y < 0 || _y >= mask.rows) continue;
                if (_x < 0 || _x >= mask.cols) continue;
                if (kernels[idx].at<uint8_t>(_y, _x) == 0) continue;
                if (mask.at<int32_t>(_y, _x) > 0) continue;

                cv::Point point_dxy(_x, _y);
                queue.push(point_dxy);

                mask.at<int32_t>(_y, _x) = value;
                is_edge = false;
            }

            if (is_edge) next_queue.push(point);
        }
        std::swap(queue, next_queue);
    }

    //contoursMap
    for (int y=0; y < mask.rows; ++y) {
        for (int x=0; x < mask.cols; ++x) {
            int idx = mask.at<int32_t>(y, x);
            if (idx == 0) continue;
            contours_map[idx].emplace_back(cv::Point(x, y));
        }
    }
}

cv::Mat matRotateClockWise180(cv::Mat src) {
	flip(src, src, 0);
    flip(src, src, 1);
	return src;
}

cv::Mat matRotateClockWise90(cv::Mat src) {
	transpose(src, src);
	flip(src, src, 1);
    return src;
}

void  OCR::detect(cv::Mat im_bgr,
                  std::vector<std::vector<cv::Point>> &bboxs,
                  std::vector<std::string> &str_predicts,
                  int long_size) {
    auto im = resize_img(im_bgr, long_size);

    float h_scale = im_bgr.rows * 1.0 / im.rows;
    float w_scale = im_bgr.cols * 1.0 / im.cols;

    ncnn::Mat in = ncnn::Mat::from_pixels(im.data, ncnn::Mat::PIXEL_BGR2RGB, im.cols, im.rows);
    in.substract_mean_normalize(mean_vals_pse_angle,norm_vals_pse_angle);

    ncnn::Extractor ex = psenet.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input("input", in);
    ncnn::Mat preds;
    double time1 = static_cast<double>( cv::getTickCount());
    ex.extract("out", preds);

    time1 = static_cast<double>( cv::getTickCount());
    std::map<int, std::vector<cv::Point>> contoursMap;
    pse_deocde(preds, contoursMap, 0.7311, 10, 1);

    std::vector<cv::RotatedRect> rects ;
    for (auto &cnt: contoursMap) {
        cv::Mat bbox;
        cv::RotatedRect rect = cv::minAreaRect(cnt.second);
        rect.size.width = rect.size.width * w_scale;
        rect.size.height = rect.size.height * h_scale;
        rect.center.x = rect.center.x * w_scale;
        rect.center.y = rect.center.y * h_scale;
        rects.push_back(rect);
        cv::boxPoints(rect, bbox);
        std::vector<cv::Point> points;
        for (int i = 0; i < bbox.rows; ++i) {
            points.emplace_back(cv::Point(int(bbox.at<float>(i, 0) ), int(bbox.at<float>(i, 1))));
        }
        bboxs.emplace_back(points);
    }

    time1 = static_cast<double>(cv::getTickCount());
    for (int i = 0; i < rects.size() ; i++) {
        //angle detection
        cv::RotatedRect temprect = rects[i];
        cv::Mat part_im;

        int  min_size = temprect.size.width>temprect.size.height?temprect.size.height:temprect.size.width;
        temprect.size.width = int(temprect.size.width + min_size * 0.15);
        temprect.size.height = int(temprect.size.height + min_size * 0.15);

        RRLib::getRotRectImg(temprect, im_bgr, part_im);

        int part_im_w = part_im.cols;
        int part_im_h = part_im.rows;
        if (part_im_h > 1.5 *  part_im_w) part_im = matRotateClockWise90(part_im);

        cv::Mat angle_input = part_im.clone();
        
        //classification
        ncnn::Mat  shufflenet_input = ncnn::Mat::from_pixels_resize(angle_input.data, 
                ncnn::Mat::PIXEL_BGR2RGB, angle_input.cols, part_im.rows ,shufflenetv2_target_w ,shufflenetv2_target_h);

        shufflenet_input.substract_mean_normalize(mean_vals_pse_angle,norm_vals_pse_angle);
        ncnn::Extractor shufflenetv2_ex = angle_net.create_extractor();
        shufflenetv2_ex.set_num_threads(num_thread);
        shufflenetv2_ex.input("input", shufflenet_input);
        ncnn::Mat angle_preds;
        double time2 = static_cast<double>( cv::getTickCount());
        shufflenetv2_ex.extract("out", angle_preds);

        float *srcdata = (float*) angle_preds.data;

        int angle_index = 0;
        int max_value;
        for (int i=0; i<angle_preds.w; i++) {
            if (i==0)
                max_value = srcdata[i];
            else if (srcdata[i] > angle_index) {
                angle_index = i;
                max_value = srcdata[i];
            }
        }
        
        if (angle_index == 0 || angle_index ==2)
            part_im = matRotateClockWise180(part_im);

        // text recognition
        int crnn_w_target ;
        float scale  = crnn_h * 1.0/ part_im.rows;
        crnn_w_target = int(part_im.cols * scale);

        cv::Mat img2 = part_im.clone();

        ncnn::Mat crnn_in = ncnn::Mat::from_pixels_resize(img2.data,
                    ncnn::Mat::PIXEL_BGR2GRAY, img2.cols, img2.rows , crnn_w_target, crnn_h);

        crnn_in.substract_mean_normalize(mean_vals_crnn,norm_vals_crnn);
       
        ncnn::Mat crnn_preds;

        if (angle_index ==0 || angle_index ==1) {

            ncnn::Extractor crnn_ex = crnn_net.create_extractor();
            crnn_ex.set_num_threads(num_thread);
            crnn_ex.input("input", crnn_in);
#if CRNN_LSTM
            // lstm
            ncnn::Mat blob162;
            crnn_ex.extract("234", blob162);

            // batch fc
            ncnn::Mat blob182(256, blob162.h);
            for (int i=0; i<blob162.h; i++) {
                ncnn::Extractor crnn_ex_1 = crnn_net.create_extractor();
                crnn_ex_1.set_num_threads(num_thread);
                ncnn::Mat blob162_i = blob162.row_range(i, 1);
                crnn_ex_1.input("253", blob162_i);

                ncnn::Mat blob182_i;
                crnn_ex_1.extract("254", blob182_i);

                memcpy(blob182.row(i), blob182_i, 256 * sizeof(float));
            }

            // lstm
            ncnn::Mat blob243;
            crnn_ex.input("260", blob182);
            crnn_ex.extract("387", blob243);

            // batch fc
            ncnn::Mat blob263(5530, blob243.h);
            for (int i=0; i<blob243.h; i++) {
                ncnn::Extractor crnn_ex_2 = crnn_net.create_extractor();
                crnn_ex_2.set_num_threads(num_thread);
                ncnn::Mat blob243_i = blob243.row_range(i, 1);
                crnn_ex_2.input("406", blob243_i);

                ncnn::Mat blob263_i;
                crnn_ex_2.extract("407", blob263_i);

                memcpy(blob263.row(i), blob263_i, 5530 * sizeof(float));
            }

            crnn_preds = blob263;
#else // CRNN_LSTM
            crnn_ex.extract("out", crnn_preds);
#endif // CRNN_LSTM
        }
        else {
            ncnn::Extractor crnn_ex = crnn_vertical_net.create_extractor();
            crnn_ex.set_num_threads(num_thread);
            crnn_ex.input("input", crnn_in);
#if CRNN_LSTM
            // lstm
            ncnn::Mat blob162;
            crnn_ex.extract("234", blob162);

            // batch fc
            ncnn::Mat blob182(256, blob162.h);
            for (int i=0; i<blob162.h; i++) {
                ncnn::Extractor crnn_ex_1 = crnn_vertical_net.create_extractor();
                crnn_ex_1.set_num_threads(num_thread);
                ncnn::Mat blob162_i = blob162.row_range(i, 1);
                crnn_ex_1.input("253", blob162_i);

                ncnn::Mat blob182_i;
                crnn_ex_1.extract("254", blob182_i);

                memcpy(blob182.row(i), blob182_i, 256 * sizeof(float));
            }

            // lstm
            ncnn::Mat blob243;
            crnn_ex.input("260", blob182);
            crnn_ex.extract("387", blob243);

            // batch fc
            ncnn::Mat blob263(5530, blob243.h);
            for (int i=0; i<blob243.h; i++) {
                ncnn::Extractor crnn_ex_2 = crnn_vertical_net.create_extractor();
                crnn_ex_2.set_num_threads(num_thread);
                ncnn::Mat blob243_i = blob243.row_range(i, 1);
                crnn_ex_2.input("406", blob243_i);

                ncnn::Mat blob263_i;
                crnn_ex_2.extract("407", blob263_i);

                memcpy(blob263.row(i), blob263_i, 5530 * sizeof(float));
            }

            crnn_preds = blob263;
#else // CRNN_LSTM
            crnn_ex.extract("out", crnn_preds);
#endif // CRNN_LSTM
        }

        auto res_pre = crnn_deocde(crnn_preds, alphabetChinese);
        std::string concat = std::accumulate(res_pre.begin(), res_pre.end(), std::string(""));
        str_predicts.push_back(concat);
    }
}
