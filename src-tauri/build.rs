fn main() {
    tauri_build::build();

    // Compile dlib C++ translation units statically using cc crate without external BLAS dependencies
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .define("DLIB_NO_GUI_SUPPORT", None)
        .include(".")
        .file("dlib/all/source.cpp")
        .file("dlib/image_processing/shape_predictor.cpp")
        .file("dlib/image_processing/shape_predictor_trainer.cpp")
        .file("cpp/dlib_wrapper.cpp")
        .warnings(false)
        .compile("dlib_core");
}
