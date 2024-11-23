#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>

#include "mpi/milovankin_m_hypercube_topology/include/ops_mpi.hpp"

namespace milovankin_m_hypercube_topology {
static void run_test_parallel(const std::string& data, int dest, std::vector<int> path_expected = {}) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;  // tests are designed for 4+ processes

  std::vector<char> data_in(data.begin(), data.end());
  std::vector<char> data_out(data_in.size());

  if (path_expected.empty()) path_expected = Hypercube::calculate_path(dest);
  std::vector<int> path_actual(std::log2(world.size()) + 1, -1);

  // Create task data
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dest));
    taskDataPar->inputs_count.emplace_back(data_in.size());
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(path_actual.data()));
    taskDataPar->outputs_count.emplace_back(data_out.size());
    taskDataPar->outputs_count.emplace_back(path_actual.size());
  }

  // Run parallel
  milovankin_m_hypercube_topology::Hypercube Hypercube(taskDataPar);
  ASSERT_TRUE(Hypercube.validation());
  ASSERT_TRUE(Hypercube.pre_processing());
  Hypercube.run();
  Hypercube.post_processing();

  // Assert
  if (world.rank() == 0) {
    path_actual.resize(path_expected.size(), -1);
    ASSERT_EQ(data_out, data_in);
    ASSERT_EQ(path_actual, path_expected);
  }
}
}  // namespace milovankin_m_hypercube_topology

TEST(milovankin_m_hypercube_topology, calculate_path_tests) {
  boost::mpi::communicator world;
  if (world.rank() != 0) return;

  std::vector<int> expect;

  expect = {0, 1};  // 0 -> 1
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(1), expect);

  expect = {0, 1, 3};  // 00 -> 01 -> 11
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(3), expect);

  expect = {0, 4};  // 00 -> 10
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(4), expect);

  expect = {0, 1, 5};  // 000 -> 001 -> 101
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(5), expect);

  expect = {0, 1, 3, 7};  // 000 -> 001 -> 011 -> 111
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(7), expect);

  expect = {0, 2, 6, 14};  // 0000 -> 0010 -> 0110 -> 1110
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(14), expect);

  expect = {0, 1, 5, 13, 29};  // 00000 -> 00001 -> 000101 -> 01101 -> 11101
  EXPECT_EQ(milovankin_m_hypercube_topology::Hypercube::calculate_path(29), expect);
}

TEST(milovankin_m_hypercube_topology, normal_input_1) {
  milovankin_m_hypercube_topology::run_test_parallel("aaabbbcccddd", 1, {0, 1});
}

TEST(milovankin_m_hypercube_topology, normal_input_2) {
  milovankin_m_hypercube_topology::run_test_parallel("Hiiii :33", 3, {0, 1, 3});
}

TEST(milovankin_m_hypercube_topology, normal_input_3) {
  milovankin_m_hypercube_topology::run_test_parallel("ABCDE", 2, {0, 2});
}

TEST(milovankin_m_hypercube_topology, large_string) {
  std::string large_str(1'000'000, 'a');
  milovankin_m_hypercube_topology::run_test_parallel(large_str, 3, {0, 1, 3});
}

TEST(milovankin_m_hypercube_topology, any_processor_count_auto_test) {
  boost::mpi::communicator world;
  int dest = world.size() / 3 * 2;
  std::vector<int> expected_path = milovankin_m_hypercube_topology::Hypercube::calculate_path(dest);
  milovankin_m_hypercube_topology::run_test_parallel("123 456 789", dest, expected_path);
}
