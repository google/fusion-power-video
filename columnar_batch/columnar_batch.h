#include "../fusion_power_video.h"

namespace fpvc {
    class BatchSchema {
    
    public:

        BatchSchema(size_t xsize, size_t ysize, size_t shifted_left, Frame &uncompressed_delta_frame);
        const size_t xsize() { return xsize_; }
        const size_t ysize() { return ysize_; }
        const size_t shiftedLeft() { return shifted_left_; }

        const std::vector<uint8_t> &compressedDeltaFrameHighPlane() { return compressed_delta_frame_high_plane_; }
        const std::vector<uint8_t> &compressedDeltaFrameLowPlane() { return compressed_delta_frame_low_plane_; }

    private:
        
        size_t xsize_;
        size_t ysize_;

        size_t shifted_left_;

        std::vector<uint8_t> compressed_delta_frame_high_plane_;
        std::vector<uint8_t> compressed_delta_frame_low_plane_;
    };

    typedef std::shared_ptr<BatchSchema> SchemaPtr;
    
    class Batch {

    public:
        Batch(size_t batch_size, SchemaPtr schema);

        void Reset();
        bool AppendPredicted(Frame predicted_frame);

        bool Empty() { return length_ == 0; }
        bool Full() { return length_ == batch_size_; }
        int64_t LatestTimestamp() { return (length_ == 0) ? -1 : timestamps_[length_-1]; }

    private:
        SchemaPtr schema_;
        size_t batch_size_;
        size_t length_;
        size_t previews_capacity_;
        size_t high_or_low_capacity_;
        std::vector<uint8_t> backing_buffer_;
        int64_t *timestamps_;

        uint8_t *flags_;
        
        uint32_t *preview_offsets_;
        uint32_t *high_plane_offsets_;
        uint32_t *low_plane_offsets_;

        uint8_t *preview_;
        uint8_t *high_plane_;
        uint8_t *low_plane_;
    };

    typedef std::shared_ptr<Batch> BatchPtr;
    
    typedef std::function<void(BatchPtr)> BatchProcessor;


}