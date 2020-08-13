#include "arrow_encoder.h"
#include <arrow/util/key_value_metadata.h>
#include <arrow/record_batch.h>
#include <iostream>

namespace fpvc {
    ArrowEncoder::ArrowEncoder(size_t xsize, size_t ysize, int shift_to_left_align, bool big_endian,
        std::function<void(std::shared_ptr<arrow::RecordBatch>)> record_batch_consumer, int frames_per_batch) 
            : record_batch_consumer_(record_batch_consumer), frames_per_batch_(frames_per_batch), xsize_(xsize), ysize_(ysize), 
            shift_to_left_align_(shift_to_left_align), big_endian_(big_endian), 
            promised_closing_timestamp_(std::promise<int64_t>()), closing_timestamp_future_(promised_closing_timestamp_.get_future()),
            promised_schema_(std::promise<std::shared_ptr<arrow::Schema>>()),
            frame_queue_(), closing_(false), delta_frame_(EMPTY),
            latestStoredTimestamp(-1), encoder_thread_(std::thread(&ArrowEncoder::EncoderTask, this, std::move(promised_schema_.get_future()))),
            timestamp_builder_(std::make_shared<arrow::Int64Builder>()), delta_predicted_builder_(std::make_shared<arrow::BooleanBuilder>()),
            cg_predicted_builder_(std::make_shared<arrow::BooleanBuilder>()), 
            preview_builder_(std::make_shared<MutableBinaryBuilder>(frames_per_batch_, frames_per_batch_ * Frame::MaxCompressedPreviewSize(xsize_, ysize_))),
            high_plane_builder_(std::make_shared<MutableBinaryBuilder>(frames_per_batch_, frames_per_batch_ * Frame::MaxCompressedPlaneSize(xsize_, ysize_))),
            low_plane_builder_(std::make_shared<MutableBinaryBuilder>(frames_per_batch_, frames_per_batch_ * Frame::MaxCompressedPlaneSize(xsize_, ysize_))) {
        
        timestamp_builder_->Reserve(frames_per_batch_);
        delta_predicted_builder_->Reserve(frames_per_batch_);
        cg_predicted_builder_->Reserve(frames_per_batch_);
    }

    ArrowEncoder::~ArrowEncoder() {
        Close();
        encoder_thread_.join();
    }

    std::future<void*> ArrowEncoder::PushFrame(uint64_t timestamp, uint16_t* frame, void* info) {
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
            std::async(std::launch::async, &ArrowEncoder::PrepareSchema, this, std::move(promised_schema_));
            // note: delta_frame_ is copied here!
            std::async(std::launch::async, &ArrowEncoder::PredictFrame, this, delta_frame_, std::move(frame_promise));
            std::promise<void*> fullfilledPromise;
            fullfilledPromise.set_value(info);
            return fullfilledPromise.get_future();
        } else {
            return std::async(std::launch::async, &ArrowEncoder::PrepareFrame, this, timestamp, frame, info, std::move(frame_promise));
        }
    }

    void* ArrowEncoder::PrepareFrame(uint64_t timestamp, uint16_t* frame, void* info, std::promise<Frame> frame_promise) {
        Frame newFrame(xsize_, ysize_, frame, shift_to_left_align_, big_endian_, timestamp);
        std::async(std::launch::async, &ArrowEncoder::PredictFrame, this, std::move(newFrame), std::move(frame_promise));
        return info;
    }

    void ArrowEncoder::PredictFrame(Frame frame, std::promise<Frame> frame_promise) {
        frame.Predict(delta_frame_);
        frame_promise.set_value(std::move(frame));
    }

    std::shared_future<int64_t> ArrowEncoder::Close() {
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

    void ArrowEncoder::PrepareSchema(std::promise<std::shared_ptr<arrow::Schema>> promised_schema) {
        Frame df = delta_frame_;
        df.Compress();
        promised_schema.set_value(arrow::schema({
                arrow::field("timestamp", arrow::timestamp(arrow::TimeUnit::NANO), false),
                arrow::field("deltaPredicted", arrow::boolean(), false),
                arrow::field("cgPredicted", arrow::boolean(), false),
                arrow::field("preview", arrow::binary(), false),
                arrow::field("highBytePlane", arrow::binary(), false),
                arrow::field("lowBytePlane", arrow::binary(), false)
            },
            arrow::key_value_metadata( {"xsize", "ysize", "shiftedLeft", "deltaFrameHighPlane", "deltaFrameLowPlane", "deltaFrameCGPredicted"},
                    {   std::to_string(xsize_), std::to_string(ysize_), std::to_string(shift_to_left_align_),
                        std::string(reinterpret_cast<const char*>(const_cast<const uint8_t*>(df.high().data())), df.high().size()),
                        std::string(reinterpret_cast<const char*>(const_cast<const uint8_t*>(df.low().data())), df.low().size()),
                        (df.flags() & FrameFlags::USE_CG) ? "true" : "false"
                        })));
    }

    void ArrowEncoder::CompressPreparedFrame(Frame frame) {
        timestamp_builder_->Append(frame.timestamp());
        delta_predicted_builder_->Append((frame.flags() & FrameFlags::USE_DELTA) > 0);
        cg_predicted_builder_->Append((frame.flags() & FrameFlags::USE_CG) > 0);

        size_t encoded_high_size = high_plane_builder_->Remaining();
        size_t encoded_low_size = low_plane_builder_->Remaining();
        size_t encoded_preview_size = preview_builder_->Remaining();

        frame.CompressPredicted(&encoded_high_size, high_plane_builder_->NextItem(), 
                &encoded_low_size, low_plane_builder_->NextItem(),
                &encoded_preview_size, preview_builder_->NextItem());

        preview_builder_->Advance(encoded_preview_size);
        high_plane_builder_->Advance(encoded_high_size);
        low_plane_builder_->Advance(encoded_low_size);
    }

    void ArrowEncoder::Flush(std::shared_future<std::shared_ptr<arrow::Schema>> schema_future) {
        const int count = timestamp_builder_->length();
        if (count == 0) {
            record_batch_consumer_(nullptr);
            return;
        }

        latestStoredTimestamp = timestamp_builder_->GetValue(count-1);

        std::shared_ptr<arrow::Array> timestamps;
        timestamp_builder_->Finish(&timestamps);
        std::shared_ptr<arrow::Array> delta_predicted;
        delta_predicted_builder_->Finish(&delta_predicted);
        std::shared_ptr<arrow::Array> cg_predicted;
        cg_predicted_builder_->Finish(&cg_predicted);

        std::shared_ptr<arrow::Array> preview;
        preview_builder_->Finish(&preview);
        std::shared_ptr<arrow::Array> high_plane;
        high_plane_builder_->Finish(&high_plane);
        std::shared_ptr<arrow::Array> low_plane;
        low_plane_builder_->Finish(&low_plane);
        
        record_batch_consumer_(arrow::RecordBatch::Make(schema_future.get(), count, {
                timestamps, delta_predicted, cg_predicted, preview, high_plane, low_plane
            }));
    }

    void ArrowEncoder::EncoderTask(std::shared_future<std::shared_ptr<arrow::Schema>> schema_future) {
        
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

                    Flush(schema_future);

                    promised_closing_timestamp_.set_value(latestStoredTimestamp);
                } else {
                    CompressPreparedFrame(std::move(frame));
                    if (timestamp_builder_->length() >= frames_per_batch_) {
                        Flush(schema_future);
                    }
                }

            }

        }
    }
}