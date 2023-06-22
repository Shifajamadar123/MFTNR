/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
 
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include "ArgusHelpers.h"
#include "CommonOptions.h"

#include <Argus/Argus.h>

#include <unistd.h>
#include <stdlib.h>
#include <sstream>
#include <iomanip>


#ifdef ANDROID
#define FILE_PREFIX "/sdcard/DCIM/"
#else
#define FILE_PREFIX ""
#endif

using namespace Argus;
/*
 * This sample opens a camera session using a sensor without ISP, and takes jpeg snapshots every second
 * using that sensor. The Jpeg saving and Preview consumption happen on a consumer thread in the
 * JPEGConsumerThread class, located in the util folder.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;


const int width = 1920;
const int height = 1080;

// Set the number of images
const int numImages = 5;



namespace ArgusSamples
{

// Debug print macros.
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)

static const uint32_t BAYER_WITHOUT_ISP_CAMERA_INDEX = 0;

static bool execute(const CommonOptions& options,int ii)
{
    const uint64_t FIVE_SECONDS_IN_NANOSECONDS = 5000000000;
    std::string s  = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/img"+std::to_string(ii)+".raw";
    char *bayerWithOutIspOutputFileName = new char[s.length()+1];
    strcpy(bayerWithOutIspOutputFileName,s.c_str());
    // Initialize the Argus camera provider.
    UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());
    //char bayerWithOutIspOutputFileName[]="new.raw";
    // Get the ICameraProvider interface from the global CameraProvider.
    ICameraProvider *iCameraProvider =
        interface_cast<ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
        ORIGINATE_ERROR("Failed to get ICameraProvider interface");
    printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    // Get the camera devices.
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() < 1)
    {
        ORIGINATE_ERROR("Insufficient camera devices.");
    }

    /****** Bayer capture without ISP ******/
    Argus::CameraDevice* bayerWithOutIspDevice = cameraDevices[BAYER_WITHOUT_ISP_CAMERA_INDEX];

    ICameraProperties *iBayerWithOutIspProperties =
         interface_cast<ICameraProperties>(bayerWithOutIspDevice);
     if (!iBayerWithOutIspProperties)
     {
         ORIGINATE_ERROR("Failed to get the JPEG camera device properties interface.");
     } 
    Argus::SensorMode* bayerWithOutIspSensorMode = ArgusSamples::ArgusHelpers::getSensorMode(
            bayerWithOutIspDevice, options.sensorModeIndex());
    // std::vector<Argus::SensorMode*> bayerWithOutIspSensorModes;
    // iBayerWithOutIspProperties->getBasicSensorModes(&bayerWithOutIspSensorModes);
    // if (!bayerWithOutIspSensorModes.size())
    // {
    //     ORIGINATE_ERROR("Failed to get valid JPEG sensor mode list.");
    // }
    ISensorMode *iBayerWithOutIspSensorMode = interface_cast<ISensorMode>(bayerWithOutIspSensorMode);
    if (!iBayerWithOutIspSensorMode)
        ORIGINATE_ERROR("Failed to get the sensor mode.");

    printf("Capturing from Resolution (%dx%d)\n",
           iBayerWithOutIspSensorMode->getResolution().width(), iBayerWithOutIspSensorMode->getResolution().height());

    // Create the preview capture session.
    UniqueObj<CaptureSession> bayerWithOutIspSession = UniqueObj<CaptureSession>(
            iCameraProvider->createCaptureSession(bayerWithOutIspDevice));
    if (!bayerWithOutIspSession)
        ORIGINATE_ERROR(
            "Failed to create preview session with camera index %d.", BAYER_WITHOUT_ISP_CAMERA_INDEX);
    ICaptureSession *iBayerWithOutIspCaptureSession = interface_cast<ICaptureSession>(bayerWithOutIspSession);
    if (!iBayerWithOutIspCaptureSession)
        ORIGINATE_ERROR("Failed to get preview capture session interface");

    // Create preview stream.
    PRODUCER_PRINT("Creating the preview stream.\n");
    UniqueObj<OutputStreamSettings> bayerWithOutIspSettings(
        iBayerWithOutIspCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
    IEGLOutputStreamSettings *iBayerWithOutIspSettings =
        interface_cast<IEGLOutputStreamSettings>(bayerWithOutIspSettings);
    if (iBayerWithOutIspSettings)
    {
        iBayerWithOutIspSettings->setPixelFormat(PIXEL_FMT_RAW16);
        iBayerWithOutIspSettings->setResolution(iBayerWithOutIspSensorMode->getResolution());
        iBayerWithOutIspSettings->setMetadataEnable(true);

    }
    UniqueObj<OutputStream> bayerWithOutIspStream(
            iBayerWithOutIspCaptureSession->createOutputStream(bayerWithOutIspSettings.get()));
    IEGLOutputStream *iBayerWithOutIspStream = interface_cast<IEGLOutputStream>(bayerWithOutIspStream);
    if (!iBayerWithOutIspStream)
        ORIGINATE_ERROR("Failed to create preview OutputStream");

    // Create the FrameConsumer to consume the output frames from the stream.
    Argus::UniqueObj<EGLStream::FrameConsumer> bayerWithOutIspConsumer(
        EGLStream::FrameConsumer::create(bayerWithOutIspStream.get()));
    EGLStream::IFrameConsumer *iBayerWithOutIspConsumer =
        Argus::interface_cast<EGLStream::IFrameConsumer>(bayerWithOutIspConsumer);
    if (!iBayerWithOutIspConsumer)
        ORIGINATE_ERROR("Failed to initialize Consumer");
        
    // Create the two requests
    UniqueObj<Request> bayerWithOutIspRequest(iBayerWithOutIspCaptureSession->createRequest());
    if (!bayerWithOutIspRequest)
        ORIGINATE_ERROR("Failed to create Request");

    IRequest *iBayerWithOutIspRequest = interface_cast<IRequest>(bayerWithOutIspRequest);
    if (!iBayerWithOutIspRequest)
        ORIGINATE_ERROR("Failed to create Request interface");

    iBayerWithOutIspRequest->setEnableIspStage(false);
    iBayerWithOutIspRequest->enableOutputStream(bayerWithOutIspStream.get());

    Argus::ISourceSettings *iBayerWithOutIspSourceSettings =
        Argus::interface_cast<Argus::ISourceSettings>(bayerWithOutIspRequest);
    if (!iBayerWithOutIspSourceSettings)
        ORIGINATE_ERROR("Failed to get source settings request interface");
    iBayerWithOutIspSourceSettings->setSensorMode(bayerWithOutIspSensorMode);
    
    // Argus is now all setup and ready to capture
    // Submit capture requests.
    PRODUCER_PRINT("Starting repeat capture requests.\n");
    
    
    uint32_t bayerWithOutIspRequestId = iBayerWithOutIspCaptureSession->capture(bayerWithOutIspRequest.get());
    if (!bayerWithOutIspRequestId)
        ORIGINATE_ERROR("Failed to submit capture request");
        
    Argus::Status status;
    Argus::UniqueObj<EGLStream::Frame> bayerWithOutIspFrame(
        iBayerWithOutIspConsumer->acquireFrame(FIVE_SECONDS_IN_NANOSECONDS, &status));
    if (status != Argus::STATUS_OK)
        ORIGINATE_ERROR("Failed to get consumer");
    EGLStream::IFrame *iBayerWithOutIspFrame = Argus::interface_cast<EGLStream::IFrame>(bayerWithOutIspFrame);
    if (!iBayerWithOutIspFrame)
        ORIGINATE_ERROR("Failed to get RGBA IFrame interface");
    EGLStream::Image *bayerWithOutIspImage = iBayerWithOutIspFrame->getImage();
    if (!bayerWithOutIspImage)
        ORIGINATE_ERROR("Failed to get RGBA Image from iFrame->getImage()");
    EGLStream::IImage *iBayerWithOutIspImage = Argus::interface_cast<EGLStream::IImage>(bayerWithOutIspImage);
    if (!iBayerWithOutIspImage)
        ORIGINATE_ERROR("Failed to get RGBA IImage");
    EGLStream::IImage2D *iBayerWithOutIspImage2D = Argus::interface_cast<EGLStream::IImage2D>(bayerWithOutIspImage);
    if (!iBayerWithOutIspImage2D)
        ORIGINATE_ERROR("Failed to get RGBA iImage2D");
    EGLStream::IImageHeaderlessFile *iBayerWithOutIspImageHeadelessFile =
        Argus::interface_cast<EGLStream::IImageHeaderlessFile>(bayerWithOutIspImage);
    if (!iBayerWithOutIspImageHeadelessFile)
        ORIGINATE_ERROR("Failed to get RGBA IImageHeaderlessFile");
    
    status = iBayerWithOutIspImageHeadelessFile->writeHeaderlessFile(bayerWithOutIspOutputFileName);
    if (status != Argus::STATUS_OK)
        ORIGINATE_ERROR("Failed to write output file");
    printf("Wrote bayerWithOutIsp file : %s\n", bayerWithOutIspOutputFileName);
    
    // Shut down Argus.
    cameraProvider.reset();
    
    PRODUCER_PRINT("Done -- exiting.\n");
    return true;
}

}// namespace ArgusSamples



void averaging()
{
    const int bitDepth = 16;

    // Allocate memory for the raw image data
    unsigned short* imageData = new unsigned short[width * height * numImages];

    // Read the raw image data from files
    for (int i = 0; i < numImages; i++) {
        std::string filename = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/img" + std::to_string(i) + ".raw";
        FILE* file=fopen64(filename.c_str(), "rb");
        fread(&imageData[i * width * height], sizeof(unsigned short), width * height, file);
        fclose(file);
    }

    // Create a vector of Mat objects to hold the image data
    vector<Mat> images(numImages);
    for (int i = 0; i < numImages; i++) {
        images[i] = Mat(height, width, CV_16UC1, &imageData[i * width * height]);
    }

    // Perform the averaging algorithm
    Mat sumImage = Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < numImages; i++) {
        Mat tempImage;
        images[i].convertTo(tempImage, CV_32FC1, 1.0 / ((1 << bitDepth) - 1));
        sumImage += tempImage;
    }
    Mat averagedImage;
    sumImage /= numImages;
    averagedImage = sumImage * ((1 << bitDepth) - 1);
    averagedImage.convertTo(averagedImage, CV_16UC1);

    // Save the averaged image as .raw format
    std::string save_filename = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/Averaged_Image.raw";
    FILE* save_file=fopen64(save_filename.c_str(), "wb");
    fwrite(averagedImage.data, sizeof(unsigned short), width * height, save_file);
    fclose(save_file);
    
    Mat colorImage(height, width, CV_16UC3);
    cv::cvtColor(averagedImage, colorImage, cv::COLOR_BayerGR2BGR);
    imwrite("/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/Averaged_image.png", colorImage);
    // Free memory
    delete[] imageData;
}



void viewRawFile()
{
    Mat final(height, width, CV_16UC3);
    final = Mat::zeros(final.size(), final.type());
    for (int i = 0; i < 3; i++)
    {
        unsigned short* imageData = new unsigned short[width * height];

        std::string filename = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/Compensated" + std::to_string(i) + ".raw";
        FILE* file=fopen64(filename.c_str(), "rb");
        fread(imageData, sizeof(unsigned short), width * height, file);
        fclose(file);
        
        Mat image(height, width, CV_16UC1, imageData);
        Mat colorImage(height, width, CV_16UC3);
        cv::cvtColor(image, colorImage, cv::COLOR_BayerGR2BGR);
        final = final + colorImage;
        delete[] imageData;
    }

    

    imwrite("/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/final.png", final);   
}


void viewSingleFile()
{
    unsigned short* imageData = new unsigned short[width * height];

    std::string filename = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/img0.raw";
    FILE* file=fopen64(filename.c_str(), "rb");
    fread(imageData, sizeof(unsigned short), width * height, file);
    fclose(file);
        
    Mat image(height, width, CV_16UC1, imageData);
    Mat colorImage(height, width, CV_16UC3);
    cv::cvtColor(image, colorImage, cv::COLOR_BayerGR2BGR);
    delete[] imageData;

    

    imwrite("/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/input_image.png", colorImage);   
}


int main(int argc, char** argv)
{
    ArgusSamples::CommonOptions options(
        basename(argv[0]),
        ArgusSamples::CommonOptions::Option_D_CameraDevice |
        ArgusSamples::CommonOptions::Option_M_SensorMode );

    if (!options.parse(argc, argv))
        return EXIT_FAILURE;
    if (options.requestedExit())
        return EXIT_SUCCESS;

    for(int i=0; i<numImages;i++){
    	if (!execute(options,i))
        	return EXIT_FAILURE;
    }
    vector<vector<Mat>> frames;// Vector of vector of Mat to store color channels for each frame

    // Read the raw image data for each frame
    for (int i = 0; i < numImages; i++) {
        std::string filename = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/img" + std::to_string(i) + ".raw";

        unsigned short* imageData = new unsigned short[width * height];
        
        FILE* file=fopen64(filename.c_str(), "rb");
        fread(imageData, sizeof(unsigned short), width * height, file);
        fclose(file);
        // Separate color channels (assuming GBRG Bayer pattern)
        // Separate color channels (assuming GBRG Bayer pattern)
        Mat redChannel(height, width, CV_16UC1);
        Mat greenChannel(height, width, CV_16UC1);
        Mat blueChannel(height, width, CV_16UC1);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if ((y % 2 == 0 && x % 2 == 0) || (y % 2 == 1 && x % 2 == 1)) {
                    greenChannel.at<unsigned short>(y, x) = imageData[y * width + x];
                    redChannel.at<unsigned short>(y, x) = 0;
                    blueChannel.at<unsigned short>(y, x) = 0;
                }
                else if (y % 2 == 0 && x % 2 == 1) {
                    greenChannel.at<unsigned short>(y, x) = 0;
                    redChannel.at<unsigned short>(y, x) = 0;
                    blueChannel.at<unsigned short>(y, x) = imageData[y * width + x];
                }
                else if (y % 2 == 1 && x % 2 == 0)
                {
                    greenChannel.at<unsigned short>(y, x) =0;
                    blueChannel.at<unsigned short>(y, x) =0;
                    redChannel.at<unsigned short>(y, x) = imageData[y * width + x];
                }
                    
                
            }

        }

        // Store color channels for each frame
        vector<Mat> frameChannels = { blueChannel, greenChannel, redChannel };

        frames.push_back(frameChannels);
        delete[] imageData;
    }

    // Define block sizes and search ranges to test
    vector<int> blockSizes = { 16 };
    vector<int> searchRanges = { 16 };


    for (int blockSize : blockSizes) {
        for (int searchRange : searchRanges) {

            for (int i = 1; i < numImages; i++) {
                // Extract color channels for current and previous frame
                vector<Mat> currentChannels = frames[i];
                vector<Mat> previousChannels = frames[i - 1];

                // Perform block matching and motion compensation for each color channel
                vector<Mat> compensatedChannels;

                for (int c = 0; c < 3; c++)
                {
                    std::string save_filename = "/home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/raw_images/Compensated"+std::to_string(c)+ ".raw";
                    Mat currFrame = currentChannels[c];
                    Mat prevFrame = previousChannels[c];

                    // Perform block matching
                    vector<Point2f> curr_points, prev_points;
                    Mat motionVectorField(currFrame.size(), CV_32FC2);

                    for (int y = 0; y < currFrame.rows - blockSize; y += blockSize)
                    {
                        for (int x = 0; x < currFrame.cols - blockSize; x += blockSize)
                        {
                            Rect currROI(x, y, blockSize, blockSize);
                            Mat currBlock = currFrame(currROI);

                            Point2f bestMatch(x, y);
                            double bestScore = numeric_limits<double>::max();

                            for (int dy = -searchRange; dy <= searchRange; dy++) {
                                for (int dx = -searchRange; dx <= searchRange; dx++) {
                                    int x_offset = x + dx;
                                    int y_offset = y + dy;

                                    if (x_offset < 0 || x_offset >= currFrame.cols - blockSize ||
                                        y_offset < 0 || y_offset >= currFrame.rows - blockSize) {
                                        continue;
                                    }

                                    Rect prevROI(x_offset, y_offset, blockSize, blockSize);
                                    Mat prevBlock = prevFrame(prevROI);

                                    double score = norm(currBlock, prevBlock, NORM_L2SQR);

                                    if (score < bestScore) {
                                        bestScore = score;
                                        bestMatch = Point2f(x_offset, y_offset);
                                    }
                                }
                            }
                            curr_points.push_back(Point2f(x + blockSize / 2, y + blockSize / 2));
                            prev_points.push_back(Point2f(bestMatch.x + blockSize / 2, bestMatch.y + blockSize / 2));

                        }
                    }

                    Point2f mean_curr_points(0.0, 0.0);
                    Point2f mean_prev_points(0.0, 0.0);
                    for (int j = 0; j < curr_points.size(); j++) {
                        mean_curr_points = mean_curr_points + curr_points[j];
                        mean_prev_points = mean_prev_points + prev_points[j];
                    }
                    mean_curr_points = mean_curr_points / (float)curr_points.size();
                    mean_prev_points = mean_prev_points / (float)prev_points.size();
                    Point2f mean_motion_vector = mean_curr_points - mean_prev_points;

                    // Define transformation matrix
                    Mat transform = (Mat_<double>(2, 3) << 1, 0, mean_motion_vector.x, 0, 1, mean_motion_vector.y);

                    // Warp previous frame to compensate for motion
                    Mat warpedPrevFrame;
                    warpAffine(prevFrame, warpedPrevFrame, transform, currFrame.size());

                    // Motion compensation and blending

                    Mat compensatedChannel;
                    addWeighted(currFrame, 0.5, warpedPrevFrame, 0.5, 0, compensatedChannel);
                    compensatedChannels.push_back(compensatedChannel);
                    
                    FILE* save_file=fopen64(save_filename.c_str(), "wb");
        	    fread(compensatedChannel.data, sizeof(unsigned short), width * height, save_file);
                    fclose(save_file);
                    
                 
                }

            }
            
        }
    } 

    viewRawFile();
    
    averaging();
    
    viewSingleFile();
    return EXIT_SUCCESS;
}
    
    
