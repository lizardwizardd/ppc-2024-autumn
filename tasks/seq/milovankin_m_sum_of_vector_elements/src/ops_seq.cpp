#include "seq/milovankin_m_sum_of_vector_elements/include/ops_seq.hpp"

namespace milovankin_m_sum_of_vector_elements_seq {

std::vector<int32_t> make_random_vector(int32_t size, int32_t val_min, int32_t val_max) {
  std::vector<int32_t> new_vector(size);

  for (int32_t i = 0; i < size; i++) {
    new_vector[i] = rand() % (val_max - val_min + 1) + val_min;
  }

  return new_vector;
}

bool VectorSumSeq::pre_processing() {
  internal_order_test();

  int32_t* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(input_ptr, input_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool VectorSumSeq::validation() {
  internal_order_test();

  return true;
}

bool VectorSumSeq::run() {
  internal_order_test();

  return true;
}

bool VectorSumSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int64_t*>(taskData->outputs[0]) = sum_;
  return true;
}

}  // namespace milovankin_m_sum_of_vector_elements_seq