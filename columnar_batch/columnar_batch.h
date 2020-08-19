#ifndef FPV_COLUMNAR_BATCH_H_
#define FPV_COLUMNAR_BATCH_H_

#include "../fusion_power_video.h"

namespace fpvc::columnarbatch {
    class BatchSchema {
    
    public:

        BatchSchema(size_t xsize, size_t ysize, size_t shifted_left, Frame &uncompressed_delta_frame);
        const size_t xsize() { return xsize_; }
        const size_t ysize() { return ysize_; }
        const size_t shiftedLeft() { return shifted_left_; }
        Frame &delta_frame() { return delta_frame_; }

        /// Delta Frame is _not_ CG predicted
        const std::vector<uint8_t> &compressedDeltaFrameHighPlane() { return compressed_delta_frame_high_plane_; }
        const std::vector<uint8_t> &compressedDeltaFrameLowPlane() { return compressed_delta_frame_low_plane_; }

    private:
        
        size_t xsize_;
        size_t ysize_;

        size_t shifted_left_;

        std::vector<uint8_t> compressed_delta_frame_high_plane_;
        std::vector<uint8_t> compressed_delta_frame_low_plane_;
        Frame delta_frame_;
    };

    typedef std::shared_ptr<BatchSchema> SchemaPtr;

    class Image {

    public:

        enum Type {
            PREVIEW,
            MSB8,
            FULL
        };

        Image(int64_t timestamp = -1, size_t xsize = 0, size_t ysize = 0, uint8_t bpp = 0,
            Type type = Type::FULL, std::vector<uint8_t> &&data = std::vector<uint8_t>());

        size_t const timestamp() { return timestamp_; }
        size_t const xsize() { return xsize_; }
        size_t const ysize() { return ysize_; }
        size_t const bpp() { return bpp_; }
        uint8_t* data8() { return data_.data(); }
        uint16_t* data16() { return reinterpret_cast<uint16_t*>(data_.data()); }
        Type const type() { return type_; }

    private:
        int64_t timestamp_;
        size_t xsize_;
        size_t ysize_;
        uint8_t bpp_;
        std::vector<uint8_t> data_;
        Type type_;
    };

    typedef std::function<void(Image)> ImageProcessor;
    
    class Batch {

    public:
        Batch(size_t batch_size, SchemaPtr schema);

        void Reset();
        bool AppendPredicted(Frame predicted_frame);

        bool Empty() { return length_ == 0; }
        bool Full() { return length_ == batch_size_; }
        int64_t LatestTimestamp() { return (length_ == 0) ? -1 : timestamps_[length_-1]; }
        size_t length() { return length_; }
        Image ExtractImage(size_t index, Image::Type type);

        SchemaPtr const schema() { return schema_; };

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

#endif // FPV_COLUMNAR_BATCH_H_
