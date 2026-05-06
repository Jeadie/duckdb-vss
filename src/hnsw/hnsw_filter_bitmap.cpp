#include "hnsw/hnsw_filter_bitmap.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"

namespace duckdb {

//! Marks row_ids from a scanned chunk that pass the given ExpressionExecutor filter into the bitmap.
//! sel is a scratch buffer; chunk has filter columns at positions 0..n-1, row_id at position n.
static void MarkMatchingRows(ExpressionExecutor &executor, DataChunk &chunk, SelectionVector &sel,
                             idx_t filter_col_count, FilterBitmap &bitmap) {
	// The executor expression references columns 0..filter_col_count-1.
	// We pass the whole chunk; the row_id at position filter_col_count is ignored.
	idx_t valid_count = executor.SelectExpression(chunk, sel);

	auto &row_id_vec = chunk.data[filter_col_count]; // last column = row_id
	// The row_id column is typically a SequenceVector synthesized by the storage engine.
	// FlatVector::GetData reads raw backing memory, which is wrong for non-flat vectors.
	// Flatten materializes the sequence in-place before we index into it.
	row_id_vec.Flatten(chunk.size());
	auto row_ids_ptr = FlatVector::GetData<row_t>(row_id_vec);

	for (idx_t i = 0; i < valid_count; i++) {
		bitmap.Set(row_ids_ptr[sel.get_index(i)]);
	}
}

unique_ptr<FilterBitmap> BuildFilterBitmap(ClientContext &context, TableCatalogEntry &table,
                                           const TableFilterSet *filters,
                                           const vector<idx_t> &logical_col_ids,
                                           const vector<LogicalType> &col_types,
                                           const vector<unique_ptr<Expression>> *extra_exprs) {
	D_ASSERT(logical_col_ids.size() == col_types.size());

	auto &storage = table.GetStorage();
	auto &transaction = DuckTransaction::Get(context, table.catalog);

	// Pre-allocate bitmap to the current total committed row count (safe upper bound for main storage).
	// Local storage row_ids can exceed this; they are handled by dynamic growth in MarkMatchingRows.
	auto bitmap = make_uniq<FilterBitmap>();
	const idx_t total_rows = storage.GetTotalRows();
	bitmap->bits.assign(total_rows, false);

	// Build scan column IDs: [filter_col_0, filter_col_1, ..., row_id]
	vector<StorageIndex> scan_col_ids;
	scan_col_ids.reserve(logical_col_ids.size() + 1);
	for (idx_t i = 0; i < logical_col_ids.size(); i++) {
		auto storage_oid = table.GetColumn(LogicalIndex(logical_col_ids[i])).StorageOid();
		scan_col_ids.emplace_back(storage_oid);
	}
	scan_col_ids.emplace_back(DConstants::INVALID_INDEX); // row_id column marker

	// Build scan types: [filter_col_types..., ROW_TYPE]
	vector<LogicalType> scan_types = col_types;
	scan_types.push_back(LogicalType::ROW_TYPE);
	const idx_t filter_col_count = col_types.size();

	// Build filter expressions, each bound to its chunk-position index (0, 1, ...) rather
	// than the original logical column ID.  This is critical: after scanning, chunk columns
	// are laid out at positions 0..n-1, independent of the source table's column numbering.
	vector<unique_ptr<Expression>> filter_exprs;

	// 1. Table-filter-derived expressions (converted from TableFilterSet, for zone-map tables).
	// IMPORTANT: Skip OPTIONAL_FILTER entries — they are hints for zone-map row-group pruning
	// only and are NOT guaranteed to be correct for row-level evaluation.  DuckDB wraps e.g.
	// id IN (10, 50, 90) in an OptionalFilter because the LogicalFilter expression
	// (id=10 OR id=50 OR id=90) is the authoritative row-level predicate.  Including the
	// OptionalFilter expression here would produce wrong bitmap marks.
	if (filters && !filters->filters.empty()) {
		filter_exprs.reserve(filters->filters.size());
		for (idx_t i = 0; i < logical_col_ids.size(); i++) {
			idx_t orig_col_id = logical_col_ids[i];
			auto it = filters->filters.find(orig_col_id);
			if (it == filters->filters.end()) {
				continue;
			}
			if (it->second->filter_type == TableFilterType::OPTIONAL_FILTER) {
				continue; // zone-map only — not safe for row-level evaluation
			}
			auto col_ref = make_uniq<BoundReferenceExpression>(col_types[i], i);
			filter_exprs.push_back(it->second->ToExpression(*col_ref));
		}
	}

	// 2. Pre-remapped LogicalFilter expressions (already use BoundReferenceExpression)
	if (extra_exprs) {
		for (const auto &expr : *extra_exprs) {
			filter_exprs.push_back(expr->Copy());
		}
	}

	if (filter_exprs.empty()) {
		// No evaluable filters; bitmap remains all-false (caller should not reach this normally)
		return bitmap;
	}

	// Combine multiple filters into a single AND expression so we can use SelectExpression
	unique_ptr<Expression> combined_expr;
	if (filter_exprs.size() == 1) {
		combined_expr = std::move(filter_exprs[0]);
	} else {
		auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : filter_exprs) {
			conjunction->children.push_back(std::move(expr));
		}
		combined_expr = std::move(conjunction);
	}

	ExpressionExecutor executor(context, *combined_expr);
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(context), scan_types);

	// Build a POSITION-remapped TableFilterSet for zone-map pruning (only when MANDATORY table
	// filters are present).  ScanFilterInfo::Initialize treats filter keys as indices into
	// scan_col_ids, NOT logical col IDs.
	//
	// IMPORTANT: Exclude OptionalFilter from zone_map_filters — if the scan engine applies an
	// OptionalFilter at the row level during Scan(), it compacts the output chunk, which offsets
	// the row_id indices used in MarkMatchingRows and produces wrong bitmap marks.  OptionalFilters
	// are safe for row-group skipping only; our ExpressionExecutor handles row-level filtering.
	TableFilterSet remapped_filters;
	if (filters && !filters->filters.empty()) {
		for (idx_t i = 0; i < logical_col_ids.size(); i++) {
			idx_t orig_col_id = logical_col_ids[i];
			auto it = filters->filters.find(orig_col_id);
			if (it == filters->filters.end()) {
				continue;
			}
			if (it->second->filter_type == TableFilterType::OPTIONAL_FILTER) {
				continue; // not safe to pass as zone_map_filters — can corrupt row-level scan
			}
			remapped_filters.filters.emplace(i, it->second->Copy());
		}
	}
	optional_ptr<TableFilterSet> zone_map_filters = remapped_filters.filters.empty()
	                                                     ? optional_ptr<TableFilterSet>()
	                                                     : optional_ptr<TableFilterSet>(&remapped_filters);

	// --- Scan main (committed) storage ---
	// Passing zone_map_filters enables row-group skipping when table_filters are present.
	// Row-level filtering is applied via ExpressionExecutor above.
	{
		TableScanState scan_state;
		storage.InitializeScan(context, transaction, scan_state, scan_col_ids, zone_map_filters);

		while (true) {
			chunk.Reset();
			storage.Scan(transaction, chunk, scan_state);
			if (chunk.size() == 0) {
				break;
			}
			MarkMatchingRows(executor, chunk, sel, filter_col_count, *bitmap);
		}
	}

	// --- Scan local (transaction-local / uncommitted) storage ---
	// Rows inserted within the current transaction are in local storage and may also
	// have been added to the HNSW index.  They must be included in the bitmap so that
	// filtered_ef_search does not exclude them.
	{
		auto &local_storage = LocalStorage::Get(context, table.catalog);
		if (local_storage.Find(storage)) {
			// TableScanState acts as the required parent for local_state (CollectionScanState).
			TableScanState local_ts;
			local_storage.InitializeScan(storage, local_ts.local_state, zone_map_filters);

			DataChunk local_chunk;
			local_chunk.Initialize(Allocator::Get(context), scan_types);

			while (true) {
				local_chunk.Reset();
				local_storage.Scan(local_ts.local_state, scan_col_ids, local_chunk);
				if (local_chunk.size() == 0) {
					break;
				}
				// Local storage row_ids may exceed the pre-allocated bitmap size; grow dynamically.
				auto &row_id_vec = local_chunk.data[filter_col_count];
				row_id_vec.Flatten(local_chunk.size());
				auto row_ids_ptr = FlatVector::GetData<row_t>(row_id_vec);
				idx_t valid_count = executor.SelectExpression(local_chunk, sel);
				for (idx_t i = 0; i < valid_count; i++) {
					row_t row_id = row_ids_ptr[sel.get_index(i)];
					if (row_id >= 0) {
						auto idx = static_cast<idx_t>(row_id);
						if (idx >= bitmap->bits.size()) {
							bitmap->bits.resize(idx + 1, false);
						}
						bitmap->bits[idx] = true;
					}
				}
			}
		}
	}

	return bitmap;
}

} // namespace duckdb
