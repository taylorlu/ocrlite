## Text detection and recognition for iOS platform

1. A simple app to demostrate how to run OCR pipeline on mobile device.
2. Use psenet to detect probable text region.
3. Support simple Chinese characters.
4. The models were quite small, it's based on NCNN framework and can build for other platforms such as Android/Windows/MacOS by cross compiler tools easily.

# Getting Started
* Build OpenCV framework for iOS platform, 
 the CMake command shell can refer to [opencv-3.4.0](https://github.com/taylorlu/opencv-3.4.0)

* Build [ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macosx-with-xcode) framework for iOS platform
 

* Build Xcode project and run the pipeline

<table>
  <tr>
    <th><img src="https://github.com/taylorlu/ocrlite/blob/master/imgs/1.jpg" height="640" width="320" ></th>
    <th><img src="https://github.com/taylorlu/ocrlite/blob/master/imgs/2.jpg" height="640" width="320" ></th>
  </tr>
  <tr>
    <th><img src="https://github.com/taylorlu/ocrlite/blob/master/imgs/3.jpg" height="640" width="320" ></th>
    <th><img src="https://github.com/taylorlu/ocrlite/blob/master/imgs/4.jpg" height="640" width="320" ></th>
  </tr>
  <tr>
    <th><img src="https://github.com/taylorlu/ocrlite/blob/master/imgs/5.jpg" height="640" width="320" ></th>
    <th><img src="https://github.com/taylorlu/ocrlite/blob/master/imgs/6.jpg" height="640" width="320" ></th>
  </tr>
</table>

# Other Text & Document algorithms
1. Document layout analysis [文档布局分析](https://github.com/taylorlu/detectron2)
2. Document Rectification and Unwarping 歪曲文档纠正 [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), [DocProj](https://github.com/xiaoyu258/DocProj)
