from pulp import *

# 1. 建立问题
prob = LpProblem("Bleding Problem", LpMinimize)

# 2. 建立变量
x1 = LpVariable("ChickenPercent", 0, None, LpInteger)
x2 = LpVariable("BeefPercent", 0)

# 3. 设置目标函数
prob += 0.013*x1 + 0.008*x2, "Total Cost of Ingredients per can"

# 4. 施加约束
prob += x1 + x2 == 100, "PercentagesSum"
prob += 0.100*x1 + 0.200*x2 >= 8.0, "ProteinRequirement"
prob += 0.080*x1 + 0.100*x2 >= 6.0, "FatRequirement"
prob += 0.001*x1 + 0.005*x2 <= 2.0, "FibreRequirement"
prob += 0.002*x2 + 0.005*x2 <= 0.4, "SaltRequirement"

# 5. 求解
solver = getSolver('PULP_CBC_CMD')
prob.solve(solver)

# 6. 打印求解状态
print("Status:", LpStatus[prob.status])

# 7. 打印出每个变量的最优值
for v in prob.variables():
    print(v.name, "=", v.varValue)

# 8. 打印最优解的目标函数值
print("Total Cost of Ingredients per can = ", value(prob.objective))