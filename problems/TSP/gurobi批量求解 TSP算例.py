import tsplib95
import numpy as np

def load_tsp(filepath):
    problem = tsplib95.load(filepath)
    # TSPLIB 原始节点编号
    node_ids = list(problem.get_nodes())
    n = len(node_ids)
    # 原始编号 -> 内部编号
    id_to_idx = {
        node_id: idx
        for idx, node_id in enumerate(node_ids)
    }
    # 内部编号 -> 原始编号
    idx_to_id = {
        idx: node_id
        for idx, node_id in enumerate(node_ids)
    }
    # cost[i, j] 表示内部节点 i 到节点 j 的成本
    cost = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                cost[i, j] = problem.get_weight(
                    idx_to_id[i],
                    idx_to_id[j]
                )
    return cost, id_to_idx, idx_to_id

# ==========================================
# 批量求解：MTZ 直接求解 / DFJ 使用 callback 动态添加约束
# ==========================================
import glob
import os
import time

import gurobipy as gp
from gurobipy import GRB

INSTANCE_DIR = r"D:\Desktop\tsp问题算例\test"
TIME_LIMIT = 300.0  # 每个算例、每种建模方式的时间限制（秒）
THREADS = 0         # 0 表示使用 Gurobi 默认线程数

STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INTERRUPTED: "INTERRUPTED",
}


def new_base_model(name, cost, lazy=False):
    """建立出度、入度约束及目标函数，返回模型、变量和边集合"""
    num_cities = cost.shape[0]
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT
    #如果为true则开启惰性约束
    if lazy:
        model.Params.LazyConstraints = 1
    #当 THREADS 等于 0 时，if THREADS: 这个条件判断为假
    if THREADS:
        model.Params.Threads = THREADS

    edges = [(i, j) for i in range(num_cities) for j in range(num_cities) if i != j]
    x = model.addVars(edges, vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(float(cost[i, j]) * x[i, j] for i, j in edges), GRB.MINIMIZE
    )
    model.addConstrs((x.sum(i, "*") == 1 for i in range(num_cities)), name="out_degree")
    model.addConstrs((x.sum("*", j) == 1 for j in range(num_cities)), name="in_degree")
    return model, x, edges


def solve_mtz(cost):
    """MTZ 约束一次性加入模型，直接求解"""
    num_cities = cost.shape[0]
    start = time.perf_counter()
    model, x, edges = new_base_model("TSP_MTZ", cost)

    # 以城市 0 为起点，其余城市引入顺序变量 u ∈ [1, n-1]
    customers = range(1, num_cities)
    u = model.addVars(customers, lb=1, ub=num_cities - 1, vtype=GRB.CONTINUOUS, name="u")
    model.addConstrs(
        (
            u[i] - u[j] + (num_cities - 1) * x[i, j] <= num_cities - 2
            for i in customers
            for j in customers
            if i != j
        ),
        name="mtz",
    )

    model.optimize()
    elapsed = time.perf_counter() - start

    status = STATUS_NAMES.get(model.Status, f"STATUS_{model.Status}")
    obj = bound = gap = None
    if model.SolCount > 0:
        obj = model.ObjVal
        bound = model.ObjBound
        gap = model.MIPGap
    return {"status": status, "time": elapsed, "obj": obj, "bound": bound, "gap": gap}


def find_subtours(num_cities, selected):
    """根据选中的边提取全部子回路"""
    succ = {i: j for i, j in selected}
    visited = [False] * num_cities
    cycles = []
    for start in range(num_cities):
        if visited[start]:
            continue
        cycle = []
        node = start
        while not visited[node]:
            visited[node] = True
            cycle.append(node)
            node = succ[node]
        cycles.append(cycle)
    return cycles


def solve_dfj(cost):
    """DFJ 约束通过 callback 动态添加：MIPSOL 时检测子回路并调用 cbLazy"""
    num_cities = cost.shape[0]
    start = time.perf_counter()
    model, x, edges = new_base_model("TSP_DFJ", cost, lazy=True)

    def subtourelim(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            vals = cb_model.cbGetSolution(x)
            selected = [(i, j) for i, j in edges if vals[i, j] > 0.5]
            for tour in find_subtours(num_cities, selected):
                if len(tour) < num_cities:
                    cb_model.cbLazy(
                        gp.quicksum(x[i, j] for i in tour for j in tour if i != j)
                        <= len(tour) - 1
                    )

    model.optimize(subtourelim)
    elapsed = time.perf_counter() - start

    status = STATUS_NAMES.get(model.Status, f"STATUS_{model.Status}")
    obj = bound = gap = None
    if model.SolCount > 0:
        obj = model.ObjVal
        bound = model.ObjBound
        gap = model.MIPGap
    return {"status": status, "time": elapsed, "obj": obj, "bound": bound, "gap": gap}


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def main():
    files = sorted(glob.glob(os.path.join(INSTANCE_DIR, "*.tsp")))
    if not files:
        print(f"未在 {INSTANCE_DIR} 下找到 .tsp 文件")
        return

    print(f"共找到 {len(files)} 个算例，每个算例每种建模方式的时间限制为 {TIME_LIMIT:.0f}s")
    print("DFJ 求解流程：callback 在 MIPSOL 中检测子回路，并通过 cbLazy 动态添加 DFJ 约束")

    results = []
    for filepath in files:
        name = os.path.basename(filepath)
        cost, _, _ = load_tsp(filepath)
        num_cities = cost.shape[0]

        print("\n" + "=" * 84)
        print(f"算例: {name}    城市数: {num_cities}")
        print("=" * 84)
        print(f"{'建模方式':<10}{'状态':<14}{'求解时间(s)':>14}{'目标值':>14}{'下界':>14}{'Gap':>10}")
        print("-" * 84)

        mtz = solve_mtz(cost)
        dfj = solve_dfj(cost)
        for label, r in (("MTZ", mtz), ("DFJ", dfj)):
            print(
                f"{label:<10}{r['status']:<14}{r['time']:>14.3f}"
                f"{fmt(r['obj'], '.2f'):>14}"
                f"{fmt(r['bound'], '.2f'):>14}"
                f"{fmt(r['gap'], '.2%'):>10}"
            )
        results.append((name, num_cities, mtz, dfj))

    print("\n" + "=" * 104)
    print("汇总（时间单位：秒）")
    print("=" * 104)
    print(
        f"{'算例':<18}{'城市数':>6}{'MTZ状态':>12}{'MTZ时间':>12}{'MTZ目标值':>14}"
        f"{'DFJ状态':>12}{'DFJ时间':>12}{'DFJ目标值':>14}"
    )
    print("-" * 104)
    for name, num_cities, mtz, dfj in results:
        print(
            f"{name:<18}{num_cities:>6}{mtz['status']:>12}{mtz['time']:>12.3f}"
            f"{fmt(mtz['obj'], '.2f'):>14}"
            f"{dfj['status']:>12}{dfj['time']:>12.3f}"
            f"{fmt(dfj['obj'], '.2f'):>14}"
        )


if __name__ == "__main__":
    main()
