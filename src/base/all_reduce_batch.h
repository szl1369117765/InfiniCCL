#ifndef INFINI_CCL_BASE_ALL_REDUCE_BATCH_H_
#define INFINI_CCL_BASE_ALL_REDUCE_BATCH_H_

#include "comm_impl.h"
#include "communicator.h"
#include "data_type_impl.h"
#include "operation.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
struct AllReduceBatchImpl;

class AllReduceBatch : public Operation<AllReduceBatch> {
 public:
  template <BackendType backend, Device::Type device>
  static ReturnStatus Execute(const void *const *send_buffs,
                              void *const *recv_buffs, const size_t *counts,
                              size_t num_ops, DataType data_type,
                              ReductionOpType op, void *comm_handle,
                              void *stream) {
    if (!comm_handle || !send_buffs || !recv_buffs || !counts || num_ops == 0 ||
        data_type < DataType::kChar || data_type >= DataType::kNumTypes ||
        op < ReductionOpType::kSum || op >= ReductionOpType::kNumRedOps) {
      return ReturnStatus::kInvalidArgument;
    }
    // Reject the whole batch before any collective can be submitted.
    for (size_t i = 0; i < num_ops; ++i) {
      if (!send_buffs[i] || !recv_buffs[i]) {
        return ReturnStatus::kInvalidArgument;
      }
    }
    return AllReduceBatchImpl<backend, device>::Apply(
        send_buffs, recv_buffs, counts, num_ops, data_type, op,
        static_cast<Communicator *>(comm_handle), stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BASE_ALL_REDUCE_BATCH_H_
