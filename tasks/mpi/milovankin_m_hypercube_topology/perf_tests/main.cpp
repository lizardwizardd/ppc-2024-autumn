#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/milovankin_m_hypercube_topology/include/ops_mpi.hpp"

TEST(milovankin_m_hypercube_topology, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;  // tests are designed for 4+ processes

  // Prepare data
  std::vector<char> data_in(16381, 'x');
  std::vector<char> data_out(data_in.size());

  int dest = world.size() - 1;
  std::vector<int> path_expected(milovankin_m_hypercube_topology::Hypercube::calculate_path(dest));
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

  // Run pipeline
  auto testMpiTaskParallel = std::make_shared<milovankin_m_hypercube_topology::Hypercube>(taskDataPar);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Assert
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    path_actual.resize(path_expected.size(), -1);
    ASSERT_EQ(data_out, data_in);
    ASSERT_EQ(path_actual, path_expected);
  }
}

TEST(milovankin_m_hypercube_topology, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;  // tests are designed for 4+ processes

  // Prepare data
  std::vector<char> data_in(16384, 'x');
  std::vector<char> data_out(data_in.size());

  int dest = world.size() - 1;
  std::vector<int> path_expected(milovankin_m_hypercube_topology::Hypercube::calculate_path(dest));
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

  // Run pipeline
  auto testMpiTaskParallel = std::make_shared<milovankin_m_hypercube_topology::Hypercube>(taskDataPar);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->task_run(perfAttr, perfResults);

  // Assert
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    path_actual.resize(path_expected.size(), -1);
    ASSERT_EQ(data_out, data_in);
    ASSERT_EQ(path_actual, path_expected);
  }
}
