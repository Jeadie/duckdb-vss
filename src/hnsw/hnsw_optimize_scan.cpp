#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

//! Returns true iff every node in the expression tree is safe to evaluate by a
//! standalone ExpressionExecutor against a plain DataChunk (no operator context).
//! Rejects subqueries and column refs that don't come from the given LogicalGet.
static bool IsExpressionSafeForPushdown(const Expression &expr, idx_t table_index,
                                        const vector<ColumnIndex> &col_ids) {
	// Subqueries require a full query-execution context — cannot evaluate standalone
	if (expr.expression_class == ExpressionClass::BOUND_SUBQUERY) {
		return false;
	}
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr.Cast<BoundColumnRefExpression>();
		// Must reference the HNSW table (not an outer or joined table)
		if (col_ref.binding.table_index != table_index) {
			return false;
		}
		// Binding index must be a valid projection position
		if (col_ref.binding.column_index >= col_ids.size()) {
			return false;
		}
		// Must not be a virtual/row-id column
		if (col_ids[col_ref.binding.column_index].GetPrimaryIndex() == DConstants::INVALID_INDEX) {
			return false;
		}
	}
	bool safe = true;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!safe) {
			return;
		}
		if (!IsExpressionSafeForPushdown(child, table_index, col_ids)) {
			safe = false;
		}
	});
	return safe;
}

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexScanOptimizer : public OptimizerExtension {
public:
	HNSWIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a LogicalGet, or on top of a LogicalFilter that
		// itself sits on top of a LogicalGet (for computed predicate pushdown).
		if (projection.children.size() != 1) {
			return false;
		}

		LogicalFilter *filter_node = nullptr;
		if (projection.children.front()->type == LogicalOperatorType::LOGICAL_FILTER) {
			filter_node = &projection.children.front()->Cast<LogicalFilter>();
			if (filter_node->children.size() != 1 ||
			    filter_node->children.front()->type != LogicalOperatorType::LOGICAL_GET) {
				return false;
			}
		} else if (projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		// get_ptr and get refer to the LogicalGet regardless of whether a LogicalFilter is in between
		auto &get_ptr = filter_node ? filter_node->children.front() : projection.children.front();
		auto &get = get_ptr->Cast<LogicalGet>();

		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// Validate all LogicalFilter expressions BEFORE any plan mutation.
		// All-or-nothing: if any expression is unsafe, leave the plan untouched.
		if (filter_node) {
			auto &col_ids = get.GetColumnIds();
			for (const auto &filter_expr : filter_node->expressions) {
				if (!IsExpressionSafeForPushdown(*filter_expr, get.table_index, col_ids)) {
					return false;
				}
			}
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		table_info.BindIndexes(context, HNSWIndex::TYPE_NAME);
		table_info.GetIndexes().Scan([&](Index &index) {
			if (!index.IsBound() || HNSWIndex::TYPE_NAME != index.GetIndexType()) {
				return false;
			}
			auto &cast_index = index.Cast<HNSWIndex>();

			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!cast_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				return false;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!cast_index.TryBindIndexExpression(get, index_expr)) {
				return false;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					return false;
				}
			}

			const auto vector_size = cast_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, cast_index, top_n.limit, std::move(query_vector));
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		// Replace the seq_scan function with the HNSW index scan function.
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = HNSWIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// If there are no filters at all (no table_filters, no LogicalFilter), remove TopN and return.
		if (get.table_filters.filters.empty() && filter_node == nullptr) {
			plan = std::move(top_n.children[0]);
			return true;
		}

		// Check whether filter pushdown into the HNSW scan is enabled
		Value pushdown_enabled_val;
		bool use_pushdown = true;
		if (context.TryGetCurrentSetting("hnsw_enable_filter_pushdown", pushdown_enabled_val)) {
			if (!pushdown_enabled_val.IsNull()) {
				use_pushdown = pushdown_enabled_val.GetValue<bool>();
			}
		}

		auto &hnsw_bind = get.bind_data->Cast<HNSWIndexScanBindData>();

		if (use_pushdown) {
			// --- Pushdown path ---

			// 1. Push table_filters into bind data (zone-map pruning + row-level evaluation)
			if (!get.table_filters.filters.empty()) {
				for (const auto &entry : get.table_filters.filters) {
					hnsw_bind.filter_logical_col_ids.push_back(entry.first);
					hnsw_bind.filter_storage_col_ids.emplace_back(
					    hnsw_bind.table.GetColumn(LogicalIndex(entry.first)).StorageOid());
					hnsw_bind.filter_col_types.push_back(get.returned_types[entry.first]);
				}
				hnsw_bind.pushed_filters = get.table_filters.Copy();
				// Clear from LogicalGet so DuckDB does not create a duplicate LogicalFilter above.
				get.table_filters.filters.clear();
			}

			// 2. Push LogicalFilter expressions: remap column refs to scan-chunk positions
			if (filter_node) {
				auto &col_ids = get.GetColumnIds();
				// Map: logical_col_id → position in filter_logical_col_ids (scan chunk position).
				// Seed with columns already registered from table_filters above.
				unordered_map<idx_t, idx_t> col_to_scan_pos;
				for (idx_t i = 0; i < hnsw_bind.filter_logical_col_ids.size(); i++) {
					col_to_scan_pos[hnsw_bind.filter_logical_col_ids[i]] = i;
				}

				// Collect all columns referenced in LogicalFilter expressions.
				// BoundColumnRefExpression.binding.column_index is a projection position;
				// GetPrimaryIndex() converts it to the logical column ID.
				for (const auto &expr : filter_node->expressions) {
					ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
					    *expr, [&](const BoundColumnRefExpression &col_ref) {
						    idx_t binding_idx    = col_ref.binding.column_index;
						    idx_t logical_col_id = col_ids[binding_idx].GetPrimaryIndex();
						    if (col_to_scan_pos.count(logical_col_id)) {
							    return; // already registered
						    }
						    idx_t scan_pos                  = hnsw_bind.filter_logical_col_ids.size();
						    col_to_scan_pos[logical_col_id] = scan_pos;
						    hnsw_bind.filter_logical_col_ids.push_back(logical_col_id);
						    hnsw_bind.filter_storage_col_ids.emplace_back(
						        hnsw_bind.table.GetColumn(LogicalIndex(logical_col_id)).StorageOid());
						    hnsw_bind.filter_col_types.push_back(get.returned_types[binding_idx]);
					    });
				}

				// Clone each expression and remap: BoundColumnRefExpression → BoundReferenceExpression.
				// Skip expressions whose column refs are ALL covered by MANDATORY (non-optional)
				// table_filters.  DuckDB may generate both a mandatory table_filter AND an
				// equivalent LogicalFilter expression; adding the LogicalFilter would double-push
				// and cause AND-combination errors.
				//
				// IMPORTANT: OptionalFilter (zone-map only, e.g. for id IN (10,50,90)) is NOT
				// mandatory — the LogicalFilter expression is the authoritative row-level predicate
				// and must NOT be skipped when the table_filter is OptionalFilter.
				for (const auto &expr : filter_node->expressions) {
					bool all_cols_in_mandatory_filters = true;
					ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
					    *expr, [&](const BoundColumnRefExpression &col_ref) {
						    if (!all_cols_in_mandatory_filters) {
							    return;
						    }
						    idx_t binding_idx    = col_ref.binding.column_index;
						    idx_t logical_col_id = col_ids[binding_idx].GetPrimaryIndex();
						    if (!hnsw_bind.pushed_filters ||
						        !hnsw_bind.pushed_filters->filters.count(logical_col_id)) {
							    all_cols_in_mandatory_filters = false;
							    return;
						    }
						    // OptionalFilter is zone-map only; not sufficient for row-level
						    auto &f = hnsw_bind.pushed_filters->filters.at(logical_col_id);
						    if (f->filter_type == TableFilterType::OPTIONAL_FILTER) {
							    all_cols_in_mandatory_filters = false;
						    }
					    });
					if (all_cols_in_mandatory_filters) {
						continue; // mandatory table_filter handles this; don't double-push
					}

					auto cloned = expr->Copy();
					ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
					    cloned,
					    [&](BoundColumnRefExpression &col_ref, unique_ptr<Expression> &expr_ptr) {
						    if (col_ref.binding.table_index != get.table_index) {
							    return;
						    }
						    idx_t logical_col_id = col_ids[col_ref.binding.column_index].GetPrimaryIndex();
						    auto it              = col_to_scan_pos.find(logical_col_id);
						    if (it != col_to_scan_pos.end()) {
							    expr_ptr =
							        make_uniq<BoundReferenceExpression>(col_ref.return_type, it->second);
						    }
					    });
					hnsw_bind.pushed_filter_expressions.push_back(std::move(cloned));
				}

				// Remove the LogicalFilter from the plan: promote the LogicalGet up to sit directly
				// under the Projection (the filter expressions are now in pushed_filter_expressions).
				projection.children.front() = std::move(filter_node->children.front());
				filter_node                 = nullptr; // pointer now dangling; clear for safety
			}

			// Pushdown complete: remove the TopN operator (HNSW scan enforces the limit).
			plan = std::move(top_n.children[0]);
			return true;
		} else {
			// --- Escape hatch: revert to post-filter behavior ---
			// Keep the TopN in the plan so LIMIT/ORDER BY are preserved above the filter.

			if (filter_node) {
				// A LogicalFilter is already in the plan.  Append any table_filter expressions to
				// it (they also need post-filtering) and leave it in place.
				auto &column_ids = get.GetColumnIds();
				for (const auto &entry : get.table_filters.filters) {
					idx_t logical_col_id = entry.first;
					auto &type           = get.returned_types[logical_col_id];
					for (idx_t i = 0; i < column_ids.size(); i++) {
						if (column_ids[i].GetPrimaryIndex() == logical_col_id) {
							auto column = make_uniq<BoundColumnRefExpression>(
							    type, ColumnBinding(get.table_index, i));
							filter_node->expressions.push_back(entry.second->ToExpression(*column));
							break;
						}
					}
				}
				filter_node->ResolveOperatorTypes();
			} else {
				// No LogicalFilter yet — create one above the LogicalGet.
				get.projection_ids.clear();
				get.types.clear();

				auto new_filter  = make_uniq<LogicalFilter>();
				auto &column_ids = get.GetColumnIds();
				for (const auto &entry : get.table_filters.filters) {
					idx_t column_id = entry.first;
					auto &type      = get.returned_types[column_id];
					bool found      = false;
					for (idx_t i = 0; i < column_ids.size(); i++) {
						if (column_ids[i].GetPrimaryIndex() == column_id) {
							column_id = i;
							found     = true;
							break;
						}
					}
					if (!found) {
						throw InternalException("Could not find column id for filter");
					}
					auto column =
					    make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
					new_filter->expressions.push_back(entry.second->ToExpression(*column));
				}
				new_filter->children.push_back(std::move(get_ptr));
				new_filter->ResolveOperatorTypes();
				get_ptr = std::move(new_filter);
			}
			// Don't remove TopN — the Filter node sits above the scan; TopN stays above everything.
			return true;
		}
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_hnsw_scan = OptimizeChildren(input.context, plan);
		if (did_use_hnsw_scan) {
			MergeProjections(plan);
		}
	}
};

//-----------------------------------------------------------------------------
// Register
//-----------------------------------------------------------------------------
void HNSWModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.optimizer_extensions.push_back(HNSWIndexScanOptimizer());
}

} // namespace duckdb
