#include "columnar_batch_decoder.h"

namespace fpvc::columnarbatch {

    ColumnarBatchDecoder::ColumnarBatchDecoder(Image::Type type, bool unshift, ImageProcessor image_processor) :
         image_processor_(image_processor), type_(type), unshift_(unshift),
         promised_closing_timestamp_(std::promise<int64_t>()), closing_timestamp_future_(promised_closing_timestamp_.get_future()),
         decoder_thread_(std::thread(&ColumnarBatchDecoder::DecoderTask, this)), batch_queue_(), closing_(false), schema_(nullptr)
         {

    }

    ColumnarBatchDecoder::~ColumnarBatchDecoder() {
        Close();
        decoder_thread_.join();
    }


    std::future<BatchPtr> ColumnarBatchDecoder::PushBatch(uint64_t timestamp, BatchPtr batch) {
        if (!schema_) {
            schema_ = batch->schema();
        }
        
        if (closing_ || (schema_.get() != batch->schema().get())) {
            return std::future<BatchPtr>();
        }
        
        PromisedBatch promised_batch(batch);
        auto future = promised_batch.future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            batch_queue_.push_back(std::move(promised_batch));
        }

        queue_condition_.notify_one();
        return future;
    }

    std::shared_future<int64_t> ColumnarBatchDecoder::Close() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            closing_ = true;
            PromisedBatch promised_batch;
            promised_batch.Done();
            batch_queue_.push_back(std::move(promised_batch));
        }
        queue_condition_.notify_one();
        return closing_timestamp_future_;
    }

    void ColumnarBatchDecoder::DecoderTask() {
        while (!(closing_ && batch_queue_.empty())) {
            PromisedBatch promised_batch;

            {
                std::unique_lock<std::mutex> lock(this->queue_mutex_);
                this->queue_condition_.wait(lock, [=]{ return closing_ || !batch_queue_.empty(); });
                if (!batch_queue_.empty()) {
                    promised_batch = std::move(batch_queue_.front());
                    batch_queue_.pop_front();
                }
            }

            if (promised_batch.batch()) {

                BatchPtr batch = promised_batch.batch();
                
                if (delta_frame_.state() == FrameState::EMPTY) {
                    delta_frame_ = Frame(schema_->xsize(), schema_->ysize(),
                        FrameFlags::NONE, FrameState::COMPRESSED, 
                        std::vector<uint8_t>(schema_->compressedDeltaFrameHighPlane()),
                        std::vector<uint8_t>(schema_->compressedDeltaFrameHighPlane()),
                        std::vector<uint8_t>());
                    delta_frame_.Uncompress();
                }

                for (size_t i = 0; i < batch->length(); i++) {
                    image_processor_(batch->ExtractImage(i, type_));
                }
                latest_provided_timestamp = batch->LatestTimestamp();
                promised_batch.Done();
            }
        }
        promised_closing_timestamp_.set_value(latest_provided_timestamp);
    }
}


