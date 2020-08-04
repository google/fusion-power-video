#include "columnar_batch_encoder.h"
#include <brotli/encode.h>
#include <iostream>

namespace fpvc {

    ColumnarBatchEncoder::ColumnarBatchEncoder(size_t xsize, size_t ysize, int shift_to_left_align, bool big_endian,
        BatchProcessor batch_processor, int frames_per_batch) 
            : batch_processor_(batch_processor), frames_per_batch_(frames_per_batch), xsize_(xsize), ysize_(ysize), 
            shift_to_left_align_(shift_to_left_align), big_endian_(big_endian), 
            promised_closing_timestamp_(std::promise<int64_t>()), closing_timestamp_future_(promised_closing_timestamp_.get_future()),
            promised_schema_(std::promise<SchemaPtr>()),
            frame_queue_(), closing_(false), delta_frame_(EMPTY),
            latest_stored_timestamp(-1), encoder_thread_(std::thread(&ColumnarBatchEncoder::EncoderTask, this, std::move(promised_schema_.get_future()))),
            current_batch_(nullptr),empty_batches_()
            {
    }

    ColumnarBatchEncoder::~ColumnarBatchEncoder() {
        Close();
        encoder_thread_.join();
    }

    std::future<void*> ColumnarBatchEncoder::PushFrame(uint64_t timestamp, uint16_t* frame, void* info) {
        auto frame_promise = std::promise<Frame>();
        {
            if (closing_) {
                return std::future<void*>();
            }
            std::unique_lock<std::mutex> lock(queue_mutex_);
            frame_queue_.push_back(frame_promise.get_future());
        }
        queue_condition_.notify_one();
        if (delta_frame_.state() == FrameState::EMPTY) {
            // handle first frame syncronously to avoid additional hassle
            delta_frame_ = Frame(xsize_, ysize_, frame, shift_to_left_align_, big_endian_, timestamp);
            std::async(std::launch::async, &ColumnarBatchEncoder::PrepareSchema, this, std::move(promised_schema_));
            // note: delta_frame_ is copied here!
            std::async(std::launch::async, &ColumnarBatchEncoder::PredictFrame, this, delta_frame_, std::move(frame_promise));
            std::promise<void*> fullfilledPromise;
            fullfilledPromise.set_value(info);
            return fullfilledPromise.get_future();
        } else {
            return std::async(std::launch::async, &ColumnarBatchEncoder::PrepareFrame, this, timestamp, frame, info, std::move(frame_promise));
        }
    }

    void* ColumnarBatchEncoder::PrepareFrame(uint64_t timestamp, uint16_t* frame, void* info, std::promise<Frame> frame_promise) {
        Frame newFrame(xsize_, ysize_, frame, shift_to_left_align_, big_endian_, timestamp);
        std::async(std::launch::async, &ColumnarBatchEncoder::PredictFrame, this, std::move(newFrame), std::move(frame_promise));
        return info;
    }

    void ColumnarBatchEncoder::PredictFrame(Frame frame, std::promise<Frame> frame_promise) {
        frame.Predict(delta_frame_);
        frame_promise.set_value(std::move(frame));
    }

    std::shared_future<int64_t> ColumnarBatchEncoder::Close() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            closing_ = true;
            auto frame_promise = std::promise<Frame>();
            frame_promise.set_value(Frame());
            frame_queue_.push_back(frame_promise.get_future());
        }
        queue_condition_.notify_one();
        return closing_timestamp_future_;
    }

    void ColumnarBatchEncoder::PrepareSchema(std::promise<SchemaPtr> promised_schema) {
        promised_schema.set_value(std::make_shared<BatchSchema>(xsize_, ysize_, shift_to_left_align_, delta_frame_));
    }

    void ColumnarBatchEncoder::Flush() {
        if ((!current_batch_) || current_batch_->Empty()) {
            std::async(std::launch::async, batch_processor_, nullptr);
            return;
        }

        latest_stored_timestamp = current_batch_->LatestTimestamp();

        std::async(std::launch::async, batch_processor_, current_batch_);
        current_batch_ = nullptr;
    }

    void ColumnarBatchEncoder::EncoderTask(std::shared_future<SchemaPtr> schema) {
        
        while (!(closing_ && frame_queue_.empty())) {
            std::shared_future<Frame> frame_future;

            {
                std::unique_lock<std::mutex> lock(this->queue_mutex_);
                this->queue_condition_.wait(lock, [=]{ return closing_ || !frame_queue_.empty(); });
                if (!frame_queue_.empty()) {
                    frame_future = std::move(frame_queue_.front());
                    frame_queue_.pop_front();
                }
            }
            if (frame_future.valid()) {
                Frame frame = frame_future.get();
                if (frame.state() == FrameState::EMPTY) {
                    if (!(closing_ && frame_queue_.empty())) {
                        // TODO: throw?
                        continue;
                    }

                    Flush();

                    promised_closing_timestamp_.set_value(latest_stored_timestamp);
                } else {
                    
                    BatchPtr batch = BatchToFill(schema);
                    batch->AppendPredicted(frame);
                    if (batch->Full()) {
                        Flush();
                    }
                }
            }
        }
    }

    BatchPtr ColumnarBatchEncoder::BatchToFill(std::shared_future<SchemaPtr> schema) {
        if ((!current_batch_) && empty_batches_.empty()) {
            current_batch_ = std::make_shared<Batch>(frames_per_batch_, schema.get());
        } else if (!current_batch_) {
            current_batch_ = empty_batches_.front();
            empty_batches_.pop_front();
        }
        return current_batch_;
    }

    void ColumnarBatchEncoder::ReturnProcessedBatch(BatchPtr processed) {
        processed->Reset();
        empty_batches_.push_back(processed);
    }
}