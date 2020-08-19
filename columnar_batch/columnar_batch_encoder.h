
#ifndef FPV_COLUMNAR_BATCH_ENCODER_H_
#define FPV_COLUMNAR_BATCH_ENCODER_H_

#include <future>
#include <list>
#include <mutex>
#include <condition_variable>
#include "columnar_batch.h"
#include "../fusion_power_video.h"

namespace fpvc::columnarbatch {
    class ColumnarBatchEncoder {
        public:

        ColumnarBatchEncoder(size_t xsize, size_t ysize, int shift_to_left_align, bool big_endian,
            BatchProcessor batch_processor, int frames_per_batch = 10);
        ~ColumnarBatchEncoder();

        std::future<void*> PushFrame(uint64_t timestamp, uint16_t* frame, void* info);

        // this should be done as a future, returne byt the BetachProcessor-call, but c++11/17 do not support
        // std::future::then or any other sensible way to handle those futures
        void ReturnProcessedBatch(BatchPtr processed);

        std::shared_future<int64_t> Close();

        private:

        void* PrepareFrame(uint64_t timestamp, uint16_t* frame, void* info, std::promise<Frame> frame_promise);
        void PredictFrame(Frame frame, std::promise<Frame> frame_promise);
        void PrepareSchema(std::promise<SchemaPtr> promised_schema);
        void Flush();

        void EncoderTask(std::shared_future<SchemaPtr> schema);
        BatchPtr BatchToFill(std::shared_future<SchemaPtr> schema);

        BatchProcessor batch_processor_;
        int frames_per_batch_;
        size_t xsize_;
        size_t ysize_;
        int shift_to_left_align_;
        bool big_endian_;
        std::promise<int64_t> promised_closing_timestamp_;
        std::shared_future<int64_t> closing_timestamp_future_;
        
        std::promise<SchemaPtr> promised_schema_;
        
        std::thread encoder_thread_;
        std::list<std::shared_future<Frame>> frame_queue_;
        std::mutex queue_mutex_;
        std::condition_variable queue_condition_;
        bool closing_;
        Frame delta_frame_;
        uint64_t latest_stored_timestamp;

        std::shared_ptr<Batch> current_batch_;
        std::list<BatchPtr> empty_batches_;

    };
}

#endif // FPV_COLUMNAR_BATCH_ENCODER_H_
