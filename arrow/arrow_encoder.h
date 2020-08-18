#include <future>
#include <list>
#include <mutex>
#include <condition_variable>
#include <arrow/type.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_binary.h>
#include "../fusion_power_video.h"

namespace fpvc {

    class ArrowEncoder {
        public:

        ArrowEncoder(size_t xsize, size_t ysize, int shift_to_left_align, bool big_endian,
            std::function<void(std::shared_ptr<arrow::RecordBatch>)> record_batch_consumer, int frames_per_batch = 10);
        ~ArrowEncoder();

        std::future<void*> PushFrame(uint64_t timestamp, uint16_t* frame, void* info);

        std::shared_future<int64_t> Close();

        private:

        void* PrepareFrame(uint64_t timestamp, uint16_t* frame, void* info, std::promise<Frame> frame_promise);
        void PredictFrame(Frame frame, std::promise<Frame> frame_promise);
        void PrepareSchema(std::promise<std::shared_ptr<arrow::Schema>> promised_schema);
        void CompressPreparedFrame(Frame frame);
        void Flush(std::shared_future<std::shared_ptr<arrow::Schema>> schema_future);

        void EncoderTask(std::shared_future<std::shared_ptr<arrow::Schema>> schema_future);

        std::function<void(std::shared_ptr<arrow::RecordBatch>)>  record_batch_consumer_;
        int frames_per_batch_;
        size_t xsize_;
        size_t ysize_;
        int shift_to_left_align_;
        bool big_endian_;
        std::promise<int64_t> promised_closing_timestamp_;
        std::shared_future<int64_t> closing_timestamp_future_;
        
        std::promise<std::shared_ptr<arrow::Schema>> promised_schema_;
        
        std::thread encoder_thread_;
        std::list<std::shared_future<Frame>> frame_queue_;
        std::mutex queue_mutex_;
        std::condition_variable queue_condition_;
        bool closing_;
        Frame delta_frame_;
        uint64_t latestStoredTimestamp;

        std::shared_ptr<arrow::Int64Builder> timestamp_builder_;
        std::shared_ptr<arrow::BooleanBuilder> delta_predicted_builder_;
        std::shared_ptr<arrow::BooleanBuilder> cg_predicted_builder_;

        class MutableBinaryBuilder {
        
        public:

            MutableBinaryBuilder(size_t max_samples = 0, size_t max_size = 0) :
                max_size_(max_size), max_samples_(max_samples) {
                Reset();
            }

            uint8_t *NextItem() { return data_pointer_; }

            size_t Remaining() { return data_buffer_->capacity() - data_buffer_->size(); }

            void Advance(size_t bytes) {
                length++;
                data_pointer_ += bytes;
                offset_pointer_[1] = offset_pointer_[0] + bytes;
                data_buffer_->Resize(offset_pointer_[1],false);
                offset_buffer_->Resize((length+1)*sizeof(arrow::BinaryType::offset_type),false);
                offset_pointer_++;
            }

            void Reset() {
                data_buffer_ = arrow::AllocateResizableBuffer(max_size_).ValueOrDie();
                offset_buffer_ = arrow::AllocateResizableBuffer((max_samples_+1) * sizeof(arrow::BinaryType::offset_type)).ValueOrDie();
                data_pointer_ = data_buffer_->mutable_data();
                offset_pointer_ = reinterpret_cast<arrow::BinaryType::offset_type*>(offset_buffer_->mutable_data());
                offset_pointer_[0] = 0;
                data_buffer_->Resize(0,false);
                offset_buffer_->Resize(sizeof(arrow::BinaryType::offset_type),false);
                length = 0;
            }

            void Finish(std::shared_ptr<arrow::Array>* out) {
                data_buffer_->Resize(offset_pointer_[0], true);
                offset_buffer_->Resize((length+1)*sizeof(arrow::BinaryType::offset_type), true);
                *out = std::make_shared<arrow::BinaryArray>(length, offset_buffer_, data_buffer_);
                Reset();
            }

        private:

            std::shared_ptr<arrow::ResizableBuffer> data_buffer_;
            std::shared_ptr<arrow::ResizableBuffer> offset_buffer_;
            uint8_t* data_pointer_;
            arrow::BinaryType::offset_type *offset_pointer_;
            size_t length;
            size_t max_size_;
            size_t max_samples_;
        };

        std::shared_ptr<MutableBinaryBuilder> preview_builder_;
        std::shared_ptr<MutableBinaryBuilder> high_plane_builder_;
        std::shared_ptr<MutableBinaryBuilder> low_plane_builder_;
    };

}


