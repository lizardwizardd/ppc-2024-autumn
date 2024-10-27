#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <vector>

#include "mpi/milovankin_m_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(milovankin_m_sum_of_vector_elements_mpi, Test_Sum_5000_Random) {
  boost::mpi::communicator world;
  std::vector<int32_t> input;
  int64_t res_par = 0;

  // Task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = milovankin_m_sum_of_vector_elements_parallel::make_random_vector(5000, -500, 500);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  ASSERT_TRUE(vectorSumPar.validation());
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  if (world.rank() == 0) {
    // Sequential
    int64_t res_seq = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
    ASSERT_TRUE(vectorSumSeq.validation());
    vectorSumSeq.pre_processing();
    vectorSumSeq.run();
    vectorSumSeq.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, regularVector) {
  boost::mpi::communicator world;
  std::vector<int32_t> input;
  int64_t res_par = 0;

  // Task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = {1, 2, 3, -5, 3, 43};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  ASSERT_TRUE(vectorSumPar.validation());
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  if (world.rank() == 0) {
    // Sequential
    int64_t res_seq = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
    ASSERT_TRUE(vectorSumSeq.validation());
    vectorSumSeq.pre_processing();
    vectorSumSeq.run();
    vectorSumSeq.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, positiveNumbers) {
  boost::mpi::communicator world;
  std::vector<int32_t> input;
  int64_t res_par = 0;

  // Task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  ASSERT_TRUE(vectorSumPar.validation());
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  if (world.rank() == 0) {
    // Sequential
    int64_t res_seq = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
    ASSERT_TRUE(vectorSumSeq.validation());
    vectorSumSeq.pre_processing();
    vectorSumSeq.run();
    vectorSumSeq.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, negativeNumbers) {
  boost::mpi::communicator world;
  std::vector<int32_t> input;
  int64_t res_par = 0;

  // Task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  ASSERT_TRUE(vectorSumPar.validation());
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  if (world.rank() == 0) {
    // Sequential
    int64_t res_seq = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
    ASSERT_TRUE(vectorSumSeq.validation());
    vectorSumSeq.pre_processing();
    vectorSumSeq.run();
    vectorSumSeq.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, zeroVector) {
  boost::mpi::communicator world;
  std::vector<int32_t> input;
  int64_t res_par = 0;

  // Task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input.resize(1000, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  ASSERT_TRUE(vectorSumPar.validation());
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  if (world.rank() == 0) {
    // Sequential
    int64_t res_seq = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
    ASSERT_TRUE(vectorSumSeq.validation());
    vectorSumSeq.pre_processing();
    vectorSumSeq.run();
    vectorSumSeq.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, emptyVector) {
  boost::mpi::communicator world;
  std::vector<int32_t> input;
  int64_t res_par = 0;

  // Task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = {};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_par));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Parallel
  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskDataPar);
  ASSERT_TRUE(vectorSumPar.validation());
  vectorSumPar.pre_processing();
  vectorSumPar.run();
  vectorSumPar.post_processing();

  if (world.rank() == 0) {
    // Sequential
    int64_t res_seq = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_seq));
    taskDataSeq->outputs_count.emplace_back(1);

    milovankin_m_sum_of_vector_elements_parallel::VectorSumSeq vectorSumSeq(taskDataSeq);
    ASSERT_TRUE(vectorSumSeq.validation());
    vectorSumSeq.pre_processing();
    vectorSumSeq.run();
    vectorSumSeq.post_processing();

    ASSERT_EQ(res_seq, res_par);
  }
}

TEST(milovankin_m_sum_of_vector_elements_mpi, validationNotPassed) {
  boost::mpi::communicator world;

  std::vector<int32_t> input = {1, 2, 3, -5};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    // Omitting output setup to cause validation to fail
  }

  milovankin_m_sum_of_vector_elements_parallel::VectorSumPar vectorSumPar(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(vectorSumPar.validation());
  }
}
