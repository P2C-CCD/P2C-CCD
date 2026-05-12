#pragma once

#include <string>

namespace p2cccd {

struct Status {
  bool ok = true;
  std::string message;

  static Status Ok() { return Status{}; }

  static Status Error(std::string msg) {
    Status status;
    status.ok = false;
    status.message = std::move(msg);
    return status;
  }
};

}  // namespace p2cccd
