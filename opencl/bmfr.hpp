#include "CLUtils/CLUtils.hpp"

#ifdef EVALUATION_MODE
#include <diffcal.hpp>
#endif // EVALUATION_MODE


// ### Edit these defines if you have different input ###
// TODO detect IMAGE_SIZES automatically from the input files
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720
// TODO detect FRAME_COUNT from the input files
#define FRAME_COUNT 60


struct ImageData {
  std::vector<cl_float> out_data[FRAME_COUNT];
  std::vector<cl_float> albedos[FRAME_COUNT];
  std::vector<cl_float> normals[FRAME_COUNT];
  std::vector<cl_float> positions[FRAME_COUNT];
  std::vector<cl_float> noisy_input[FRAME_COUNT];

#ifdef EVALUATION_MODE
  std::vector<cl_float> reference_data[FRAME_COUNT];
#endif
};


#ifdef EVALUATION_MODE

struct ImageDataIterator : public ImageIterator {
  ImageData* image_data;
  int start_value, curr;

public:

  ImageDataIterator(int start_value, ImageData& image_data,
		    int width, int height) {
    this->image_data = &image_data;

    this->start_value = start_value;
    this->curr = start_value;

    
    this->width = width;
    this->height = height;
  }

  void reset() {
    this->curr = start_value;
  }
  
  virtual float *getImage1() {
    return image_data->out_data[this->curr].data();
  }

  virtual float *getImage2() {
    return image_data->reference_data[this->curr].data();
  }

  virtual float *getLast() {
    return image_data->out_data[this->curr - 1].data();
  }

  virtual bool hasLast() {
    return curr != this->start_value;
  }

  virtual bool forward() {
    this->curr++;
    if(this->curr == FRAME_COUNT) {
      return false;
    }
    return true;
  }
};

#endif // EVALUATION_MODE
