#ifndef INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_REDUCE_BATCH_H_
#define INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_REDUCE_BATCH_H_

#include "backends/ccl/common/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/all_reduce_batch.h"
#include "communicator.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
class CclAllReduceBatchImpl {
 public:
  static ReturnStatus Apply(const void *const *send_buffs,
                            void *const *recv_buffs, const size_t *counts,
                            size_t num_ops, DataType data_type,
                            ReductionOpType op, Communicator *comm,
                            void *stream) {
    using Api = CclApi<backend, device>;
    using TypeMap = CclTypeMap<backend, device>;
    using CommInstance = CclCommInstance<Api>;
    if (!comm || !comm->intra_comm() || comm->intra_comm_backend() != backend ||
        comm->device_type() != device) {
      return ReturnStatus::kInternalError;
    }
    auto *instance = static_cast<CommInstance *>(comm->intra_comm());
    if (!instance->handle) {
      return ReturnStatus::kInternalError;
    }
    typename Api::DataType native_type{};
    typename Api::RedOp native_op{};
    if (!TypeMap::ToBackendDataType(data_type, &native_type) ||
        !TypeMap::ToBackendRedOp(op, &native_op)) {
      return ReturnStatus::kNotSupported;
    }

    const auto status = Api::Check(Api::GroupStart());
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    ReturnStatus first_error = ReturnStatus::kSuccess;
    for (size_t i = 0; i < num_ops; ++i) {
      const auto result = Api::Check(Api::AllReduce(
          send_buffs[i], recv_buffs[i], counts[i], native_type, native_op,
          instance->handle, reinterpret_cast<typename Api::Stream>(stream)));
      if (first_error == ReturnStatus::kSuccess &&
          result != ReturnStatus::kSuccess) {
        first_error = result;
      }
    }
    // Match the existing native AllToAll rule: close every started group.
    const auto end_status = Api::Check(Api::GroupEnd());
    return first_error != ReturnStatus::kSuccess ? first_error : end_status;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_COMMON_IMPL_ALL_REDUCE_BATCH_H_
