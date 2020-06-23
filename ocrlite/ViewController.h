//
//  ViewController.h
//  ocrlite
//
//  Created by ludong on 2019/5/22.
//  Copyright © 2019 ludong. All rights reserved.
//

#include "ocr.h"
#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <CoreGraphics/CGContext.h>

@interface ViewController : UIViewController<AVCaptureVideoDataOutputSampleBufferDelegate>{
    
    __weak IBOutlet UIImageView *imageView;
    __weak IBOutlet UITextView *textView;
    
    uint8_t *originData;
    uint8_t *planerData;
    
    AVCaptureVideoDataOutput *output;
    AVCaptureSession     *session;
    AVCaptureDeviceInput *inputDevice;
    AVCaptureVideoPreviewLayer   *previewLayer;
    
    AVCaptureDevice *backCamera;
    
    OCR *ocrengine;
}


@end

