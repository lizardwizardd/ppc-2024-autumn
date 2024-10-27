#include "mpi/milovankin_m_sum_of_vector_elements/include/ops_mpi.hpp"

namespace milovankin_m_sum_of_vector_elements_parallel {

std::vector<int32_t> make_random_vector(int32_t size, int32_t val_min, int32_t val_max) {
  std::vector<int32_t> new_vector(size);

  for (int32_t i = 0; i < size; i++) {
    new_vector[i] = rand() % (val_max - val_min + 1) + val_min;
  }

  return new_vector;
}

//
// Sequential version
//

bool VectorSumSeq::validation() {
  internal_order_test();

  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorSumSeq::pre_processing() {
  internal_order_test();

  // Fill input vector from taskData
  int32_t* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(input_ptr, input_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool VectorSumSeq::run() {
  internal_order_test();

  sum_ = 0;
  for (int32_t num : input_) {
    sum_ += num;
  }

  return true;
}

bool VectorSumSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int64_t*>(taskData->outputs[0]) = sum_;
  return true;
}

//
// Parallel version
//

bool VectorSumPar::validation() {
  internal_order_test();
  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorSumPar::pre_processing() {
  internal_order_test();

  int my_rank = world.rank();
  int world_size = world.size();
  int total_size = 0;

  // Fill input vector in root process
  if (my_rank == 0) {
    total_size = taskData->inputs_count[0];

    input_.resize(total_size);
    int32_t* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    std::copy(input_ptr, input_ptr + total_size, input_.begin());
  }

  boost::mpi::broadcast(world, total_size, 0);

  // Determine local vector size
  int local_size = total_size / world_size;
  int remainder = total_size % world_size;

  if (my_rank < remainder) {
    local_size += 1;
  }

  input_.resize(local_size);

  // Divide input vector among processes
  // Vectors are calculated in root process
  std::vector<int> send_counts, offsets;
  if (my_rank == 0) {
    send_counts.resize(world_size, total_size / world_size);
    for (int i = 0; i < remainder; ++i) {
      send_counts[i]++;
    }

    offsets.resize(world_size, 0);
    for (int i = 1; i < world_size; ++i) {
      offsets[i] = offsets[i - 1] + send_counts[i - 1];
    }
  }

  boost::mpi::scatterv(world, reinterpret_cast<int32_t*>(taskData->inputs[0]), send_counts, offsets, input_.data(),
                       local_size, 0);

  return true;
}

bool VectorSumPar::run() {
  internal_order_test();

  int64_t local_sum = std::accumulate(input_.begin(), input_.end(), int64_t(0));
  boost::mpi::reduce(world, local_sum, sum_, std::plus<int64_t>(), 0);

  return true;
}

bool VectorSumPar::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    *reinterpret_cast<int64_t*>(taskData->outputs[0]) = sum_;
  }

  return true;
}

}  // namespace milovankin_m_sum_of_vector_elements_parallel
