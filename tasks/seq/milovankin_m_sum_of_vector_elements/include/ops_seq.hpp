#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_sum_of_vector_elements_seq {

[[nodiscard]] std::vector<int32_t> make_random_vector(int32_t size, int32_t minVal, int32_t maxVal);

class VectorSumSeq : public ppc::core::Task {
 public:
  explicit VectorSumSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_;
  int64_t sum_ = 0;
};

}  // namespace milovankin_m_sum_of_vector_elements_seq