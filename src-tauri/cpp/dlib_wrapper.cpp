#include "dlib_wrapper.h"

#ifndef DLIB_NO_GUI_SUPPORT
#define DLIB_NO_GUI_SUPPORT
#endif

#include "dlib/image_processing/shape_predictor.h"
#include "dlib/image_processing/shape_predictor_trainer.h"
#include "dlib/data_io/load_image_dataset.h"
#include "dlib/array2d.h"
#include "dlib/pixel.h"
#include "dlib/serialize.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <exception>

extern "C" {

int dlib_predict_landmarks(
    const char* predictor_path,
    const uint8_t* img_data,
    int width,
    int height,
    int channels,
    const DlibRectangle* bbox,
    DlibPoint2D* out_points,
    size_t max_points,
    size_t* out_num_points
) {
    if (!predictor_path || !img_data || !out_points || !out_num_points || width <= 0 || height <= 0) {
        return -1;
    }

    try {
        dlib::shape_predictor predictor;
        dlib::deserialize(predictor_path) >> predictor;

        dlib::array2d<dlib::rgb_pixel> img(height, width);
        if (channels == 3) {
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    size_t idx = (r * width + c) * 3;
                    img[r][c] = dlib::rgb_pixel(img_data[idx], img_data[idx + 1], img_data[idx + 2]);
                }
            }
        } else if (channels == 1) {
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    size_t idx = r * width + c;
                    uint8_t val = img_data[idx];
                    img[r][c] = dlib::rgb_pixel(val, val, val);
                }
            }
        } else {
            return -2; // Unsupported channels
        }

        dlib::rectangle rect;
        if (bbox && bbox->width > 0 && bbox->height > 0) {
            rect = dlib::rectangle(
                bbox->left,
                bbox->top,
                bbox->left + bbox->width - 1,
                bbox->top + bbox->height - 1
            );
        } else {
            rect = dlib::rectangle(0, 0, width - 1, height - 1);
        }

        dlib::full_object_detection shape = predictor(img, rect);
        size_t num_parts = shape.num_parts();
        *out_num_points = num_parts;

        size_t copy_count = (num_parts < max_points) ? num_parts : max_points;
        for (size_t i = 0; i < copy_count; ++i) {
            out_points[i].x = shape.part(i).x();
            out_points[i].y = shape.part(i).y();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "dlib_predict_landmarks error: " << e.what() << std::endl;
        return -3;
    } catch (...) {
        return -4;
    }
}

int dlib_train_predictor(
    const char* xml_dataset_path,
    const char* output_model_path,
    unsigned long oversampling_amount,
    double nu,
    unsigned long tree_depth,
    unsigned long num_trees_per_cascade_level
) {
    if (!xml_dataset_path || !output_model_path) {
        return -1;
    }

    try {
        dlib::array<dlib::array2d<dlib::rgb_pixel>> images;
        std::vector<std::vector<dlib::full_object_detection>> objects;

        dlib::load_image_dataset(images, objects, xml_dataset_path);

        dlib::shape_predictor_trainer trainer;
        trainer.set_oversampling_amount(oversampling_amount);
        trainer.set_nu(nu);
        trainer.set_tree_depth(tree_depth);
        trainer.set_num_trees_per_cascade_level(num_trees_per_cascade_level);

        dlib::shape_predictor predictor = trainer.train(images, objects);

        dlib::serialize(output_model_path) << predictor;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "dlib_train_predictor error: " << e.what() << std::endl;
        return -2;
    } catch (...) {
        return -3;
    }
}

} // extern "C"
