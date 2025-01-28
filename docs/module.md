## 模型

1. 考虑每个运动员的贡献，并假设 2028 年参赛运动员与 2024 年一致

2. 运动员的动量为国家动量、项目动量、个人动量的加权求和，其中国家动量和项目动量归为初始动量

$$
M_i^{(t)} = a_1 \cdot C_i^{(t_0)} + a_2 \cdot S_i^{(t_0)} + a_3 \cdot P_i^{(t)}
$$

3. 假设历史表现更好的国家派出的选手质量更高，国家动量根据所在国家队的历史表现，考虑国家队过去得奖情况的指数移动平均；过去得奖情况指当年获得各个奖牌**占比**的加权求和 ÷ 总参赛选手，分S母固定正偏置，可训练正系数；分子系数、偏置均可训练

$$
C_j^{(t)} = \alpha^{(t)} c_j^{(t)} + (1 - \alpha^{(t)}) C_j^{(t-1)} \\
\alpha^{(t)} = \frac{2}{t + 1},\ t \ge 1 \\
c_j^{(t)} = \frac{1}{1 + \beta_0 N} \left[
    \beta_1 \cdot \text{Gold} + \beta_2 \cdot \text{Silver} + \beta_3 \cdot \text{Bronze} + 1
\right]
$$

注：这里的牌子数为获奖占比，$N$ 为参赛人数

4. 类似的，计算项目初始动量

$$
S_j^{(t)} = \alpha^{(t)} s_j^{(t)} + (1 - \alpha^{(t)}) S_j^{(t-1)} \\
\alpha^{(t)} = \frac{2}{t + 1},\ t \ge 1 \\
s_j^{(t)} = \frac{1}{1 + \gamma_0 N} \left[
    \gamma_1 \cdot \text{Gold} + \gamma_2 \cdot \text{Silver} + \gamma_3 \cdot \text{Bronze} + 1
\right]
$$

5. 初始动量不随时间变化，即在运动员首次参赛时确定

6. 对于没有历史表现的国家 / 项目，初始动量由偏置决定；如有参赛但没获奖的，正常计算

$$
C_j^{(1)} = c_j^{(1)} = 1
$$

7. 动态动量由个人的历史表现、是否主办方决定；每年、每个项目，所有运动员的动态动量会被标准化（0-1，σ=1）

$$
P_i^{'(t)} = P_i^{'(t-1)} + \eta_1 \cdot \text{Gold} + \eta_2 \cdot \text{Silver} + \eta_3 \cdot \text{Bronze} + \eta_4 \cdot \text{No Medal} + \eta_5 \cdot \text{Host} \\
P_i^{(t)} = \frac{P_i^{'(t)} - \mu^{(t)}}{\sigma^{(t)}}
$$

注：这里的牌子数为获奖数目

8. 对于没有历史表现的运动员，动态动量初始为 0

9. 每个项目每个运动员的获奖能力 θ 用 softmax 计算，若金牌数为 1，则获金牌的期望即 θ；获得奖牌数的期望为 min(1, θ * 总奖牌数)；若金牌数大于 1，则类似总奖牌数计算；银牌、铜牌期望作差即可

$$
\Theta_i^{(t)} = \frac{e^{M_i^{(t)}}}{\sum_{j=1}^{N} e^{M_j^{(t)}}} \\
\text{Gold} = \min \left( 1, \Theta_i^{(t)} \cdot \text{Total Gold} \right) \\
\text{Silver} = \min \left( 1, \Theta_i^{(t)} \cdot \text{Total Silver and Gold} \right) - \text{Gold} \\
\text{Bronze} = \min \left( 1, \Theta_i^{(t)} \cdot \text{Total Medal} \right) - \text{Silver} - \text{Gold} \\
\text{Total Medal} = \text{Total Gold} + \text{Total Silver} + \text{Total Bronze}
$$

10. 实际奖牌分配有更多需要处理的细节以真实模拟实际情况，详见代码