import pandas as pd
import argparse
from scipy.stats import ttest_ind

def analyze_ab_test(control_path, treatment_path, alpha=0.05):
    """
    执行A/B实验的统计分析。
    
    :param control_path: 对照组数据的CSV文件路径。
    :param treatment_path: 实验组数据的CSV文件路径。
    :param alpha: 显著性水平，默认为0.05。
    """
    try:
        # --- 1. 数据加载 ---
        control_df = pd.read_csv(control_path)
        treatment_df = pd.read_csv(treatment_path)
    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        return

    # --- 2. 计算核心指标 ---
    # 样本量
    n_control = len(control_df)
    n_treatment = len(treatment_df)

    # 转化人数
    conv_control_count = control_df['converted'].sum()
    conv_treatment_count = treatment_df['converted'].sum()

    # 转化率 (Conversion Rate, CVR)
    cvr_control = conv_control_count / n_control
    cvr_treatment = conv_treatment_count / n_treatment
    
    # 转化率差异
    cvr_diff = cvr_treatment - cvr_control
    # 相对提升 (Relative Lift)
    if cvr_control > 0:
        cvr_lift = cvr_diff / cvr_control
    else:
        cvr_lift = float('inf') # 如果对照组转化率为0，则提升为无穷大

    # --- 3. 执行独立样本t检验 ---
    # ttest_ind 用于计算两个独立样本的t检验。
    # equal_var=False 执行 Welch's t-test，它不假设两个总体的方差相等，是更稳健的选择。
    stat, p_value = ttest_ind(treatment_df['converted'], control_df['converted'], equal_var=False)

    # --- 4. 打印格式化的分析报告 ---
    print("\n" + "="*40)
    print("      A/B 实验评估分析报告")
    print("="*40 + "\n")
    
    print(f"{'指标':<20} | {'对照组 (Control)':<20} | {'实验组 (Treatment)':<20}")
    print("-"*65)
    print(f"{'样本量 (N)':<20} | {n_control:<20} | {n_treatment:<20}")
    print(f"{'转化人数':<20} | {conv_control_count:<20} | {conv_treatment_count:<20}")
    print(f"{'转化率 (CVR)':<20} | {cvr_control:<20.4f} | {cvr_treatment:<20.4f}")
    print("-"*65 + "\n")

    print("--- 实验效果分析 ---\n")
    print(f"转化率绝对差异: {cvr_diff:+.4f}")
    print(f"转化率相对提升: {cvr_lift:+.2%}\n")
    
    print("--- 统计显著性检验 (Welch's t-test) ---\n")
    print(f"t-statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4f}\n")

    # --- 5. 得出结论 ---
    print("--- 结论 ---\n")
    if p_value < alpha:
        print(f"结果在统计上是显著的 (p < {alpha})。")
        print("我们可以拒绝原假设，认为实验组和对照组之间存在真实差异。")
        if cvr_diff > 0:
            print("实验策略带来了显著的正向提升。")
        else:
            print("实验策略带来了显著的负向影响。")
    else:
        print(f"结果在统计上不显著 (p >= {alpha})。")
        print("我们没有足够的证据拒绝原假设，不能认为两组之间存在真实差异。")
        print("实验策略带来的效果可能是由随机波动引起的。")
    
    print("\n" + "="*40 + "\n")

def main():
    # --- 使用 argparse 解析命令行参数 ---
    # argparse 是Python标准库中用于创建命令行工具的模块。
    parser = argparse.ArgumentParser(description="A/B 实验评估工具")
    
    # 添加两个必需的位置参数
    parser.add_argument("control_path", help="对照组CSV文件的路径")
    parser.add_argument("treatment_path", help="实验组CSV文件的路径")
    
    args = parser.parse_args()
    
    # 调用核心分析函数
    analyze_ab_test(args.control_path, args.treatment_path)

if __name__ == "__main__":
    main()