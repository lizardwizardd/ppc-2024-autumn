#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_hypercube_topology {
class Hypercube : public ppc::core::Task {
 public:
  struct DataIn {
    std::vector<int> path;
    std::vector<char> data;

    int destination;

    DataIn() = default;
    DataIn(const std::string& str, int dest) : data(str.begin(), str.end()), destination(dest) {}

    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar & destination;
      ar & data;
      ar & path;
    }
  };

 public:
  explicit Hypercube(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  DataIn data_;
};

}  // namespace milovankin_m_hypercube_topology