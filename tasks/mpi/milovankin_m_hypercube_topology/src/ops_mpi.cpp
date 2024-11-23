#include "mpi/milovankin_m_hypercube_topology/include/ops_mpi.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <vector>

bool milovankin_m_hypercube_topology::Hypercube::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs.size() != 2) return false;
    if (taskData->outputs.empty() || taskData->outputs_count.size() != 2) return false;
  }

  return true;
}

bool milovankin_m_hypercube_topology::Hypercube::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* dataPtr = reinterpret_cast<char*>(taskData->inputs[0]);
    data_.data.resize(taskData->inputs_count[0]);
    std::copy(dataPtr, dataPtr + taskData->inputs_count[0], data_.data.begin());

    data_.destination = *reinterpret_cast<int*>(taskData->inputs[1]);

    data_.path.clear();
  }

  return true;
}

bool milovankin_m_hypercube_topology::Hypercube::run() {
  internal_order_test();

  int world_size = world.size();
  int my_rank = world.rank();

  auto getNextId = [&world_size, &my_rank, &dest = this->data_.destination]() {
    for (uint16_t i = 0; i < std::log2(world_size); ++i) {
      uint16_t next = my_rank ^ (1 << i);
      if ((next ^ dest) < (my_rank ^ dest)) {
        return (int)next;
      }
    }
    return -1;  // supposed to never happen
  };

  if (world.rank() == 0) {  // source process
    data_.path.push_back(0);
    int next = getNextId();
    if (next == -1) return false;
    world.send(next, 0, data_);
    world.recv(boost::mpi::any_source, 0, data_);

    // Send termination signal to unused processes
    data_.path[0] = -1;
    for (int i = 1; i < world.size(); ++i) {
      if (std::find(data_.path.begin(), data_.path.end(), i) == data_.path.end()) {
        world.send(i, 0, data_);
      }
    }
    data_.path[0] = 0;

  } else {
    // Recieve data, finish if it contains termination signal
    world.recv(boost::mpi::any_source, 0, data_);
    if (data_.path[0] == -1) return true;

    data_.path.push_back(world.rank());
    if (world.rank() != data_.destination) {  // intermediate process, calculate next and send
      int next = getNextId();
      if (next == -1) return false;
      world.send(next, 0, data_);
    } else {
      world.send(0, 0, data_);  // destination reached, send back to source process
    }
  }

  return true;
}

bool milovankin_m_hypercube_topology::Hypercube::post_processing() {
  internal_order_test();
  world.barrier();

  if (world.rank() == 0) {
    auto* data_out_ptr = reinterpret_cast<char*>(taskData->outputs[0]);
    std::copy(data_.data.begin(), data_.data.end(), data_out_ptr);

    auto* path_out_ptr = reinterpret_cast<int*>(taskData->outputs[1]);
    std::copy(data_.path.begin(), data_.path.end(), path_out_ptr);
  }

  return true;
}

// Calculate expected path from 0 to destination
std::vector<int> milovankin_m_hypercube_topology::Hypercube::calculate_path(int dest) {
  std::vector<int> path = {0};

  int current = 0;
  for (uint16_t i = 0; i <= std::log2(dest); ++i) {
    uint16_t next = current ^ (1 << i);  // flip i-th bit
    if ((next ^ dest) < (current ^ dest)) {
      path.push_back(next);
      current = next;
    }
  }

  return path;
}
