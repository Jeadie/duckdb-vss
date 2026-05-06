#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/storage_index.hpp"

namespace duckdb {

class ClientContext;
class TableCatalogEntry;

//! A dense boolean bitmap indexed by row_t. Used to pre-filter HNSW searches.
//! Rows not present in the bitmap are excluded from graph traversal so they do
//! not consume the `wanted` budget, avoiding post-filter underfill.
struct FilterBitmap {
	vector<bool> bits; // bit-packed: 1 bit per row_id slot

	bool Contains(row_t row_id) const noexcept {
		if (row_id < 0) {
			return false;
		}
		auto idx = static_cast<idx_t>(row_id);
		return idx < bits.size() && bits[idx];
	}

	void Set(row_t row_id) {
		if (row_id < 0) {
			return;
		}
		auto idx = static_cast<idx_t>(row_id);
		if (idx < bits.size()) {
			bits[idx] = true;
		}
		// row_ids beyond the pre-allocated size are silently ignored
		// (they originate from concurrent inserts after GetTotalRows())
	}
};

//! Scan the filter columns of `table` for all transaction-visible rows that
//! satisfy `filters` and mark their row_ids in the returned bitmap.
//!
//! @param logical_col_ids  Logical column indices that are keys in filters.filters
//! @param col_types        Types of those columns (positionally aligned)
unique_ptr<FilterBitmap> BuildFilterBitmap(ClientContext &context, TableCatalogEntry &table,
                                           const TableFilterSet &filters,
                                           const vector<idx_t> &logical_col_ids,
                                           const vector<LogicalType> &col_types);

} // namespace duckdb
