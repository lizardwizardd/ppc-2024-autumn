#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <vector>

#include "mpi/milovankin_m_sum_of_vector_elements/include/ops_mpi.hpp"

// Run parallel, return validation result
bool run_parallel_sum(std::vector<int32_t>& input, boost::mpi::communicator& world, int64_t& res_par) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  if (!vectorSumPar.validation()) return false;
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  return true;
}

// Run sequential, return validation result
bool run_sequential_sum(std::vector<int32_t>& input, int64_t& res_seq) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res_seq));
  taskDataSeq->outputs_count.emplace_back(1);

  milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
  if (!vectorSumSeq.validation()) return false;
  vectorSumSeq.pre_processing();
  vectorSumSeq.run();
  vectorSumSeq.post_processing();

  return true;
}

// Run parallel and sequential, then compare the results
void run_parallel_vs_sequential_test(std::vector<int32_t> input) {
  boost::mpi::communicator world;
  int64_t res_par = 0;
  ASSERT_TRUE(run_parallel_sum(input, world, res_par));

  if (world.rank() == 0) {
    int64_t res_seq = 0;
    ASSERT_TRUE(run_sequential_sum(input, res_seq));
    ASSERT_EQ(res_seq, res_par);
  }
}

// Tests using helper functions
TEST(milovankin_m_sum_of_vector_elements_mpi, Test_Sum_5000_Random) {
  if (boost::mpi::communicator().rank() == 0) {
    auto input = milovankin_m_sum_of_vector_elements_parallel::make_random_vector(5000, -500, 500);
    run_parallel_vs_sequential_test(input);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, regularVector) {
  std::vector<int32_t> input = {1, 2, 3, -5, 3, 43};
  run_parallel_vs_sequential_test(input);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, positiveNumbers) {
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  run_parallel_vs_sequential_test(input);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, negativeNumbers) {
  std::vector<int32_t> input = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  run_parallel_vs_sequential_test(input);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, zeroVector) {
  std::vector<int32_t> input(1000, 0);
  run_parallel_vs_sequential_test(input);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, emptyVector) {
  std::vector<int32_t> input = {};
  run_parallel_vs_sequential_test(input);
}

TEST(milovankin_m_sum_of_vector_elements_mpi, validationNotPassed) {
  boost::mpi::communicator world;

  std::vector<int32_t> input = {1, 2, 3, -5};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    // Omitting output setup to cause validation to fail
  }

  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(vectorSumPar.validation());
  }
}
