//
//  ViewController.m
//  ocrlite
//
//  Created by ludong on 2019/5/22.
//  Copyright © 2019 ludong. All rights reserved.
//
#import "ViewController.h"

using namespace cv;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    char *path = (char *)[[[NSBundle mainBundle] resourcePath] UTF8String];
    ocrengine = new OCR(path);

    [textView setEditable:NO];
    [self startCapture];
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {

    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;

    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }

    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );

    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return finalImage;
}

cv::Mat draw_bbox(cv::Mat src, const std::vector<std::vector<cv::Point>> bboxs) {
    cv::Mat dst;
    if (src.channels() == 1) {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
    } else {
        dst = src.clone();
    }
    auto color = cv::Scalar(255, 0, 0);
    for (auto bbox :bboxs) {
        cv::line(dst, bbox[0], bbox[1], color, 1);
        cv::line(dst, bbox[1], bbox[2], color, 1);
        cv::line(dst, bbox[2], bbox[3], color, 1);
        cv::line(dst, bbox[3], bbox[0], color, 1);
    }
    return dst;
}

//cv::Mat draw_text(cv::Mat textImage, std::vector<std::vector<cv::Point>> bboxs, std::vector<std::string> str_predicts) {
//    int font_face = cv::FONT_HERSHEY_COMPLEX;
//    double font_scale = 0.3;
//    int thickness = 1;
//    int baseline;
//    int i = 0;
//
//    for(std::vector<std::string>::iterator iter=str_predicts.begin(); iter!=str_predicts.end(); iter++, i++) {
//        std::string text = *iter;
//        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
//        cv::Point point = bboxs[i][3];
//        cv::putText(textImage, text, point, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
//    }
//
//    return textImage;
//}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection  {
    
    [connection setVideoOrientation:AVCaptureVideoOrientationPortrait];
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    int width = (int)CVPixelBufferGetWidth(imageBuffer);
    int height = (int)CVPixelBufferGetHeight(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    
    if(planerData==NULL) {
        planerData = (uint8_t *)malloc(width*height*3);
    }
    int cnt = 0;
    int planeSize = width*height;
    for(int i=0; i<width*height; i++) {
        planerData[planeSize*2 + cnt] = baseAddress[i*4];
        planerData[planeSize + cnt] = baseAddress[i*4+1];
        planerData[cnt] = baseAddress[i*4+2];
        cnt++;
    }

    cv::Mat chn[] = {
        cv::Mat(height, width, CV_8UC1, planerData),  // starting at 1st blue pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize),    // 1st green pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize*2)   // 1st red pixel
    };
    // RGB --> BGR
    cv::Mat frame;
    merge(chn, 3, frame);
    
    std::vector<std::vector<cv::Point>> bboxs;
    std::vector<std::string> str_predicts;
    ocrengine->detect(frame, bboxs, str_predicts);

    frame = draw_bbox(frame, bboxs);
    
    NSString *text = @"";
    for(std::vector<std::string>::iterator iter=str_predicts.begin(); iter!=str_predicts.end(); iter++) {
        text = [text stringByAppendingString:[[NSString alloc] initWithUTF8String:iter->c_str()]];
        text = [text stringByAppendingString:@"\n"];
    }

    dispatch_async(dispatch_get_main_queue(), ^{
        [imageView setImage:[self UIImageFromCVMat:frame]];
        [textView setText:text];
    });
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
}

-(void)startCapture {
    
    NSArray *cameraArray = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in cameraArray) {
        [device lockForConfiguration:nil];
        [device setActiveVideoMaxFrameDuration:CMTimeMake(1, 1)];
        [device setActiveVideoMinFrameDuration:CMTimeMake(1, 10)];
        [device unlockForConfiguration];
        
        if ([device position] == AVCaptureDevicePositionBack) {
            backCamera = device;
        }
    }

    session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPreset640x480;
    inputDevice = [AVCaptureDeviceInput deviceInputWithDevice:backCamera error:nil];
    [session addInput:inputDevice];     //输入设备与session连接
    
    /*  设置输出yuv格式   */
    output = [[AVCaptureVideoDataOutput alloc] init];
    NSNumber *value = [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA];
    NSDictionary *dictionary = [NSDictionary dictionaryWithObject:value forKey:(NSString *)kCVPixelBufferPixelFormatTypeKey];
    [output setVideoSettings:dictionary];
    [output setAlwaysDiscardsLateVideoFrames:YES];
    
    /*  设置输出回调队列    */
    dispatch_queue_t queue = dispatch_queue_create("com.linku.queue", NULL);
    [output setSampleBufferDelegate:self queue:queue];
    //    dispatch_release(queue);
    [session addOutput:output];     //输出与session连接

    planerData = NULL;

    [session startRunning];
}

@end
