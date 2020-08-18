#include <iostream>
#include <memory>
#include "columnar_batch_encoder.h"
#include "columnar_batch_decoder.h"

std::unique_ptr<fpvc::columnarbatch::ColumnarBatchEncoder> encoder;
std::unique_ptr<fpvc::columnarbatch::ColumnarBatchDecoder> decoder;

void decodeRecordBatch(fpvc::columnarbatch::BatchPtr batch) {
    if (batch) {
        std::cout << "Got the Batch! " << std::endl;
        encoder->ReturnProcessedBatch(decoder->PushBatch(batch).get());
        std::cout << "Returned the Batch! " << batch->LatestTimestamp() << std::endl;
    } else {
        std::cout << "Got the NULLPTR!" << std::endl;
    }
}

void compareImage(fpvc::columnarbatch::Image image) {
    static uint16_t ii = 1;
    std::cout << "Got the Image " << ii << "! " << image.timestamp() << std::endl;
    for (uint16_t i=0; i<100*100;i++) {
        if (image.data16()[i] != i*ii)
            std::cout << "Bad Pixel " << i << " (" << image.data16()[i] << " != " << (i*ii) << ")" << std::endl;
    }
    ii++;
}

int main() {
    std::cout << "FPV Arrow Encoder Test" << std::endl;
    encoder = std::make_unique<fpvc::columnarbatch::ColumnarBatchEncoder>(100,100,0,false,&decodeRecordBatch,2);
    decoder = std::make_unique<fpvc::columnarbatch::ColumnarBatchDecoder>(fpvc::columnarbatch::Image::Type::FULL, false, compareImage);

    uint16_t img[100*100];
    for (uint16_t i=0; i<100*100;i++) img[i] = i;

    std::cout << "Pushing frame." << std::endl;
    encoder->PushFrame(123456,img,img).wait();

    std::cout << "Pushing second frame." << std::endl;
    for (uint16_t i=0; i<100*100;i++) img[i] += i;
    encoder->PushFrame(234567,img,img).wait();


    std::cout << "Pushing third frame." << std::endl;
    for (uint16_t i=0; i<100*100;i++) img[i] += i;
    encoder->PushFrame(345678,img,img).wait();

    std::cout << "Frame buffer available again." << std::endl;

    auto end = encoder->Close().get();
    std::cout << "Closed Encoder - " << end << "." << std::endl;

    auto dend = decoder->Close().get();
    std::cout << "Closed Decoder - " << dend << "." << std::endl;
}
