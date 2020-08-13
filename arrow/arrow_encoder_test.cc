#include <iostream>
#include "arrow_encoder.h"
#include <arrow/record_batch.h>

void printRecordBatch(std::shared_ptr<arrow::RecordBatch> batch) {
    if (batch) {
        std::cout << "Got the Batch! \n" << batch->ToString() << std::endl;
        std::cout << "    Schema: \n" <<   batch->schema()->ToString(true) << std::endl;
    } else {
        std::cout << "Got the NULLPTR!" << std::endl;
    }
}

void printRecordBatch2(std::shared_ptr<arrow::RecordBatch> batch) {
    if (batch) {
        std::cout << "Got the Batch!" << batch->RemoveColumn(3).ValueUnsafe()->RemoveColumn(3).ValueUnsafe()->RemoveColumn(3).ValueUnsafe()->ToString() << std::endl;
    } else {
        std::cout << "Got the NULLPTR!" << std::endl;
    }
}

int main() {
    std::cout << "FPV Arrow Encoder Test" << std::endl;
    fpvc::ArrowEncoder encoder(100,100,0,false,&printRecordBatch,2);
    uint16_t img[100*100];
    std::cout << "Pushing frame." << std::endl;
    encoder.PushFrame(123456,img,img).wait();
    std::cout << "Pushing second frame." << std::endl;
    encoder.PushFrame(234567,img,img).wait();
    std::cout << "Frame buffer available again." << std::endl;
    auto end = encoder.Close().get();
    std::cout << "Closed - " << end << "." << std::endl;

    std::cout << "Second part" << std::endl;
    fpvc::ArrowEncoder encoder2(100,100,0,false,&printRecordBatch2,130);
    for (int i=0;i<500;i++) {
        uint16_t img2[100*100];
        img2[0] = i;
        for (int j=1;j<100*100;j++) {
            img2[j] = std::rand();
        }
        std::cout << "Pushing frame " << std::to_string(i) << std::endl;
        std::shared_future<void*> fut = encoder2.PushFrame(i,img2,img2);
        std::async(std::launch::async, [](std::shared_future<void*> fut_) { fut_.get(); std::cout << "Buffer is back!" << std::endl;}, std::move(fut));
    }
    auto end2 = encoder2.Close().get();
    std::cout << "Closed - " << end2 << "." << std::endl;
}