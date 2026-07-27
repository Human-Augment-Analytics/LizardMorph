#ifndef DLIB_WRAPPER_H
#define DLIB_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 2D Point structure for landmark coordinates
typedef struct {
    double x;
    double y;
} DlibPoint2D;

// Bounding box structure (left, top, width, height)
typedef struct {
    long left;
    long top;
    long width;
    long height;
} DlibRectangle;

/**
 * @brief Predict landmark coordinates for an image crop/region using a trained dlib shape predictor (.dat).
 * 
 * @param predictor_path Path to the .dat shape predictor file (null-terminated string).
 * @param img_data Pointer to image pixel array (RGB, BGR, or Grayscale).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param channels Number of channels (1 = grayscale, 3 = RGB/BGR).
 * @param bbox Pointer to bounding box rectangle within the image. If null, entire image is used.
 * @param out_points Array allocated by caller to receive predicted DlibPoint2D coordinates.
 * @param max_points Capacity of out_points array.
 * @param out_num_points Output pointer to receive the actual number of predicted landmarks.
 * @return int 0 on success, non-zero error code on failure.
 */
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
);

/**
 * @brief Train a dlib shape predictor model from XML annotations and save to output file (.dat).
 * 
 * @param xml_dataset_path Path to dlib XML dataset file (null-terminated string).
 * @param output_model_path Path where trained model (.dat) will be saved.
 * @param oversampling_amount Oversampling factor for training (e.g. 20).
 * @param nu Regularization parameter (e.g. 0.1).
 * @param tree_depth Depth of regression trees (e.g. 4).
 * @param num_trees_per_cascade_level Trees per cascade level (e.g. 500).
 * @return int 0 on success, non-zero error code on failure.
 */
int dlib_train_predictor(
    const char* xml_dataset_path,
    const char* output_model_path,
    unsigned long oversampling_amount,
    double nu,
    unsigned long tree_depth,
    unsigned long num_trees_per_cascade_level
);

#ifdef __cplusplus
}
#endif

#endif // DLIB_WRAPPER_H
