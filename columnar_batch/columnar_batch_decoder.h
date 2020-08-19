#ifndef FPV_COLUMNAR_BATCH_DECODER_H_
#define FPV_COLUMNAR_BATCH_DECODER_H_

#include <future>
#include <list>
#include <mutex>
#include <condition_variable>
#include "columnar_batch.h"
#include "../fusion_power_video.h"

namespace fpvc::columnarbatch {
    class ColumnarBatchDecoder {
        public:

        ColumnarBatchDecoder(Image::Type type, bool unshift, ImageProcessor image_processor);
        ~ColumnarBatchDecoder();

        std::future<BatchPtr> PushBatch(BatchPtr batch);

        std::shared_future<int64_t> Close();

        private:

        void DecoderTask();

        ImageProcessor image_processor_;
        Image::Type type_;
        bool unshift_;
        std::promise<int64_t> promised_closing_timestamp_;
        std::shared_future<int64_t> closing_timestamp_future_;
        
        class PromisedBatch {
        public:
            PromisedBatch(const PromisedBatch &other) = delete; // not copyable as promise is not; auto-deletes copy-assign 
            PromisedBatch(PromisedBatch &&other) = default; // also auto-deleted
            PromisedBatch& operator= (PromisedBatch&& other) = default; // also auto-deleted

            PromisedBatch(BatchPtr batch = nullptr) : batch_(batch) {}

            void Done() { promise_.set_value(batch_); }
            BatchPtr const batch() { return batch_; }
            std::future<BatchPtr> future() { return promise_.get_future(); }
        
        private:

            BatchPtr batch_;
            std::promise<BatchPtr> promise_;
        };

        std::thread decoder_thread_;
        std::list<PromisedBatch> batch_queue_;
        std::mutex queue_mutex_;
        std::condition_variable queue_condition_;
        bool closing_;
        Frame delta_frame_;
        SchemaPtr schema_;
        uint64_t latest_provided_timestamp;
    };
}

#endif // FPV_COLUMNAR_BATCH_DECODER_H_
