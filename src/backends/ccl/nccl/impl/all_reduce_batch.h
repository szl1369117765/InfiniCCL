#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_REDUCE_BATCH_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_REDUCE_BATCH_H_

#include "backends/ccl/common/impl/all_reduce_batch.h"

namespace infini::ccl {

template <Device::Type device>
class AllReduceBatchImpl<BackendType::kNccl, device>
    : public CclAllReduceBatchImpl<BackendType::kNccl, device> {};

template <>
struct BackendEnabled<AllReduceBatch, BackendType::kNccl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_IMPL_ALL_REDUCE_BATCH_H_
