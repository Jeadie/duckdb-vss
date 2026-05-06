#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

//! Returns true iff every node in the expression tree is safe to evaluate by a
//! standalone ExpressionExecutor against a plain DataChunk (no operator context).
//! Rejects subqueries and column refs that don't come from the given LogicalGet.
static bool IsExpressionSafeForTopKPushdown(const Expression &expr, idx_t table_index,
                                            const vector<ColumnIndex> &col_ids) {
	if (expr.expression_class == ExpressionClass::BOUND_SUBQUERY) {
		return false;
	}
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr.Cast<BoundColumnRefExpression>();
		if (col_ref.binding.table_index != table_index) {
			return false;
		}
		if (col_ref.binding.column_index >= col_ids.size()) {
			return false;
		}
		if (col_ids[col_ref.binding.column_index].GetPrimaryIndex() == DConstants::INVALID_INDEX) {
			return false;
		}
	}
	bool safe = true;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!safe) {
			return;
		}
		if (!IsExpressionSafeForTopKPushdown(child, table_index, col_ids)) {
			safe = false;
		}
	});
	return safe;
}

//------------------------------------------------------------------------------
// Optimizer Helpers
//------------------------------------------------------------------------------

static unique_ptr<Expression> CreateListOrderByExpr(ClientContext &context, unique_ptr<Expression> elem_expr,
                                                    unique_ptr<Expression> order_expr,
                                                    unique_ptr<Expression> filter_expr) {
	auto func_entry =
	    Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, "", "", "list", OnEntryNotFound::RETURN_NULL);
	if (!func_entry) {
		return nullptr;
	}

	auto func = func_entry->functions.GetFunctionByOffset(0);
	vector<unique_ptr<Expression>> arguments;
	arguments.push_back(std::move(elem_expr));

	auto agg_bind_data = func.bind(context, func, arguments);
	auto new_agg_expr =
	    make_uniq<BoundAggregateExpression>(func, std::move(arguments), std::move(std::move(filter_expr)),
	                                        std::move(agg_bind_data), AggregateType::NON_DISTINCT);

	// We also need to order the list items by the distance
	BoundOrderByNode order_by_node(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(order_expr));
	new_agg_expr->order_bys = make_uniq<BoundOrderModifier>();
	new_agg_expr->order_bys->orders.push_back(std::move(order_by_node));

	return std::move(new_agg_expr);
}

//------------------------------------------------------------------------------
// Main Optimizer
//------------------------------------------------------------------------------
// This optimizer rewrites
//
//	AGG(MIN_BY(t1.col1, distance_func(t1.col2, query_vector), k)) <- TABLE_SCAN(t1)
//  =>
//	AGG(LIST(col1 ORDER BY distance_func(col2, query_vector) ASC)) <- HNSW_INDEX_SCAN(t1, query_vector, k)
//

class HNSWTopKOptimizer : public OptimizerExtension {
public:
	HNSWTopKOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a Aggregate operator
		if (plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			return false;
		}
		// Look for a expression that is a distance expression
		auto &agg = plan->Cast<LogicalAggregate>();
		if (!agg.groups.empty() || agg.expressions.size() != 1) {
			return false;
		}

		auto &agg_expr = agg.expressions[0];
		if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
			return false;
		}
		auto &agg_func_expr = agg_expr->Cast<BoundAggregateExpression>();
		if (agg_func_expr.function.name != "min_by") {
			return false;
		}
		if (agg_func_expr.children.size() != 3) {
			return false;
		}
		if (agg_func_expr.children[2]->type != ExpressionType::VALUE_CONSTANT) {
			return false;
		}
		const auto &col_expr   = agg_func_expr.children[0];
		const auto &dist_expr  = agg_func_expr.children[1];
		const auto &limit_expr = agg_func_expr.children[2];

		// we need the aggregate to be on top of a single child
		if (agg.children.size() != 1) {
			return false;
		}

		// The aggregate may sit directly on a LogicalGet, or on a LogicalFilter → LogicalGet
		// (for computed predicate pushdown).
		LogicalFilter *filter_node = nullptr;
		if (agg.children[0]->type == LogicalOperatorType::LOGICAL_FILTER) {
			filter_node = &agg.children[0]->Cast<LogicalFilter>();
			if (filter_node->children.size() != 1 ||
			    filter_node->children.front()->type != LogicalOperatorType::LOGICAL_GET) {
				return false;
			}
		} else if (agg.children[0]->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = filter_node ? filter_node->children.front() : agg.children[0];
		auto &get     = get_ptr->Cast<LogicalGet>();

		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// Validate all LogicalFilter expressions BEFORE any plan mutation.
		if (filter_node) {
			auto &col_ids = get.GetColumnIds();
			for (const auto &filter_expr : filter_node->expressions) {
				if (!IsExpressionSafeForTopKPushdown(*filter_expr, get.table_index, col_ids)) {
					return false;
				}
			}
		}

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

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
			if (!cast_index.TryMatchDistanceFunction(dist_expr, bindings)) {
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

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
			    !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					return false;
				}
			}

			const auto vector_size    = cast_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;

			auto query_vector    = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}
			const auto k_limit = limit_expr->Cast<BoundConstantExpression>().value.GetValue<int32_t>();
			if (k_limit <= 0 || k_limit >= STANDARD_VECTOR_SIZE) {
				return false;
			}
			bind_data =
			    make_uniq<HNSWIndexScanBindData>(duck_table, cast_index, k_limit, std::move(query_vector));
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		// Replace the aggregate with a index scan + projection
		get.function = HNSWIndexScanFunction::GetFunction();
		const auto cardinality          = get.function.cardinality(context, bind_data.get());
		get.has_estimated_cardinality   = cardinality->has_estimated_cardinality;
		get.estimated_cardinality       = cardinality->estimated_cardinality;
		get.bind_data                   = std::move(bind_data);

		// Replace the aggregate with a list() aggregate function ordered by the distance
		agg.expressions[0] = CreateListOrderByExpr(context, col_expr->Copy(), dist_expr->Copy(),
		                                           agg_func_expr.filter ? agg_func_expr.filter->Copy() : nullptr);

		// If there are no filters at all, nothing more to do.
		if (get.table_filters.filters.empty() && filter_node == nullptr) {
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

			// 1. Push table_filters into bind data
			if (!get.table_filters.filters.empty()) {
				for (const auto &entry : get.table_filters.filters) {
					hnsw_bind.filter_logical_col_ids.push_back(entry.first);
					hnsw_bind.filter_storage_col_ids.emplace_back(
					    hnsw_bind.table.GetColumn(LogicalIndex(entry.first)).StorageOid());
					hnsw_bind.filter_col_types.push_back(get.returned_types[entry.first]);
				}
				hnsw_bind.pushed_filters = get.table_filters.Copy();
				get.table_filters.filters.clear();
			}

			// 2. Push LogicalFilter expressions: remap column refs to scan-chunk positions
			if (filter_node) {
				auto &col_ids = get.GetColumnIds();
				unordered_map<idx_t, idx_t> col_to_scan_pos;
				for (idx_t i = 0; i < hnsw_bind.filter_logical_col_ids.size(); i++) {
					col_to_scan_pos[hnsw_bind.filter_logical_col_ids[i]] = i;
				}

				for (const auto &expr : filter_node->expressions) {
					ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
					    *expr, [&](const BoundColumnRefExpression &col_ref) {
						    idx_t binding_idx    = col_ref.binding.column_index;
						    idx_t logical_col_id = col_ids[binding_idx].GetPrimaryIndex();
						    if (col_to_scan_pos.count(logical_col_id)) {
							    return;
						    }
						    idx_t scan_pos                  = hnsw_bind.filter_logical_col_ids.size();
						    col_to_scan_pos[logical_col_id] = scan_pos;
						    hnsw_bind.filter_logical_col_ids.push_back(logical_col_id);
						    hnsw_bind.filter_storage_col_ids.emplace_back(
						        hnsw_bind.table.GetColumn(LogicalIndex(logical_col_id)).StorageOid());
						    hnsw_bind.filter_col_types.push_back(get.returned_types[binding_idx]);
					    });
				}

				// Skip expressions whose column refs are ALL covered by MANDATORY (non-optional)
				// table_filters.  DuckDB may generate both a mandatory table_filter AND an
				// equivalent LogicalFilter expression; adding the LogicalFilter would double-push.
				//
				// OptionalFilter (zone-map only, e.g. id IN (10,50,90)) is NOT mandatory —
				// the LogicalFilter expression is the authoritative row-level predicate.
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

				// Remove the LogicalFilter: promote LogicalGet up to sit directly under Aggregate.
				agg.children[0] = std::move(filter_node->children.front());
				filter_node     = nullptr;
			}
		} else {
			// --- Escape hatch ---

			if (filter_node) {
				// LogicalFilter already in plan. Append table_filter expressions to it.
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
		}

		return true;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		if (!TryOptimize(input.optimizer.binder, input.context, plan)) {
			// Recursively optimize the children
			for (auto &child : plan->children) {
				Optimize(input, child);
			}
		}
	}
};

void HNSWModule::RegisterTopKOptimizer(DatabaseInstance &db) {
	// Register the TopKOptimizer
	db.config.optimizer_extensions.push_back(HNSWTopKOptimizer());
}

} // namespace duckdb
