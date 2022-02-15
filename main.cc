#include <treelite/tree.h>
#include <treelite/gtil.h>
#include <treelite/frontend.h>
#include <xgboost/c_api.h>
#include <iostream>
#include <string>
#include <random>
#include <memory>
#include <limits>
#include <chrono>
#include <algorithm>
#include <exception>
#include <cstddef>

void xgb_check(int err) {
  if (err != 0) {
    throw std::runtime_error(std::string{XGBGetLastError()});
  }
}

class RandomGenerator {
 public:
  RandomGenerator() : rng_(std::random_device()()), real_dist_(0.0, 1.0) {}

  float DrawReal(float low, float high) {
    return real_dist_(rng_) * (high - low) + low;
  }

 private:
  std::mt19937 rng_;
  std::uniform_real_distribution<float> real_dist_;
};

struct DenseData {
  float* data;
  std::size_t num_row;
  std::size_t num_col;

  DenseData(std::size_t num_row, std::size_t num_col) : num_row(num_row), num_col(num_col) {
    data = new float[num_row * num_col];
  }
  ~DenseData() {
    delete [] data;
  }
};

std::unique_ptr<DenseData> GenerateData(std::size_t num_row, std::size_t num_feature) {
  std::unique_ptr<DenseData> ret = std::make_unique<DenseData>(num_row, num_feature);
  RandomGenerator rng;
  std::generate_n(ret->data, num_row * num_feature, [&rng]() { return rng.DrawReal(-1.0, 1.0); });
  return ret;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " {gtil,xgb} [XGBoost model path] [num_feature] "
              << "[number of rows to test]" << std::endl;
    return 1;
  }

  std::string runtime{argv[1]};
  std::string model_path{argv[2]};
  std::size_t num_feature = static_cast<std::size_t>(std::stol(std::string{argv[3]}));
  std::size_t num_row = static_cast<std::size_t>(std::stol(std::string{argv[4]}));

  std::unique_ptr<DenseData> input = GenerateData(num_row, num_feature);

  std::chrono::high_resolution_clock::time_point tstart, tend;
  if (std::string(runtime) == "gtil") {
    std::cout << "GTIL" << std::endl;
    std::unique_ptr<treelite::Model> model
        = treelite::frontend::LoadXGBoostJSONModel(model_path.c_str());

    tstart = std::chrono::high_resolution_clock::now();
    float* output = new float[treelite::gtil::GetPredictOutputSize(model.get(), num_row)];

    treelite::gtil::Predict(model.get(), input->data, num_row, output, -1, true);

    delete [] output;
    tend = std::chrono::high_resolution_clock::now();
  } else if (std::string(runtime) == "xgb") {
    std::cout << "XGBoost" << std::endl;
    BoosterHandle bst;
    DMatrixHandle dmat;
    xgb_check(XGBoosterCreate(nullptr, 0, &bst));
    xgb_check(XGBoosterLoadModel(bst, model_path.c_str()));

    tstart = std::chrono::high_resolution_clock::now();
    xgb_check(XGDMatrixCreateFromMat(input->data, num_row, num_feature,
              std::numeric_limits<float>::quiet_NaN(), &dmat));
    const float* out_result = nullptr;
    bst_ulong out_size = 0;

    xgb_check(XGBoosterPredict(bst, dmat, 0, 0, 0, &out_size, &out_result));

    xgb_check(XGDMatrixFree(dmat));
    tend = std::chrono::high_resolution_clock::now();

    xgb_check(XGBoosterFree(bst));
  } else {
    std::cerr << "Unrecognized choice for runtime: " << runtime << std::endl;
    return 2;
  }
  std::cout << "Time elapsed: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms"
      << std::endl;

  return 0;
}
