import pandas as pd
import argparse
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import confint_proportions_2indep

def analyze_ab_test(control_path, treatment_path, alpha=0.05):
    """
    执行A/B实验的统计分析，并输出格式化的报告。
    
    :param control_path: 对照组数据的CSV文件路径。
    :param treatment_path: 实验组数据的CSV文件路径。
    :param alpha: 显著性水平，默认为0.05。
    """
    try:
        control_df = pd.read_csv(control_path)
        treatment_df = pd.read_csv(treatment_path)
    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        return

    # --- 1. 计算核心指标 ---
    n_control, n_treatment = len(control_df), len(treatment_df)
    conv_control_count = control_df['converted'].sum()
    conv_treatment_count = treatment_df['converted'].sum()
    cvr_control = conv_control_count / n_control
    cvr_treatment = conv_treatment_count / n_treatment
    cvr_diff = cvr_treatment - cvr_control
    cvr_lift = cvr_diff / cvr_control if cvr_control > 0 else float('inf')

    # --- 2. 执行统计检验 ---
    stat, p_value = ttest_ind(treatment_df['converted'], control_df['converted'], equal_var=False)
    
    # --- 3. 计算置信区间 ---
    # 使用 statsmodels 计算转化率差异的置信区间
    ci_low, ci_high = confint_proportions_2indep(
        count1=conv_treatment_count, nobs1=n_treatment,
        count2=conv_control_count, nobs2=n_control,
        method='wald', alpha=alpha
    )

    # --- 4. 打印格式化的分析报告 ---
    print("\n" + "="*60)
    print(" " * 20 + "A/B 实验评估分析报告")
    print("="*60)
    
    print("\n" + "[ 核心指标 ]".center(60, "-"))
    print(f"{'':<5}{'指标':<20} | {'对照组 (Control)':<15} | {'实验组 (Treatment)':<15}")
    print("-" * 60)
    print(f"{'':<5}{'样本量 (N)':<20} | {n_control:<15} | {n_treatment:<15}")
    print(f"{'':<5}{'转化人数':<20} | {conv_control_count:<15} | {conv_treatment_count:<15}")
    print(f"{'':<5}{'转化率 (CVR)':<20} | {f'{cvr_control:.2%}':<15} | {f'{cvr_treatment:.2%}':<15}")
    print("-" * 60)

    print("\n" + "[ 效果评估 ]".center(60, "-"))
    print(f" ▸ 转化率绝对差异: {cvr_diff:+.2%}")
    print(f" ▸ 转化率相对提升: {cvr_lift:+.2%}")
    print(f" ▸ 差异置信区间 (95%): [{ci_low:+.2%}, {ci_high:+.2%}]")
    
    print("\n" + "[ 统计显著性 ]".center(60, "-"))
    print(f" ▸ Welch's t-test p-value: {p_value:.4f}")

    print("\n" + "[ 实验结论 ]".center(60, "-"))
    if p_value < alpha:
        print(f" ✅ 结果在统计上是显著的 (p < {alpha})。")
        if cvr_diff > 0:
            print("    我们可以相信，实验策略带来了真实且显著的正向提升。")
        else:
            print("    我们可以相信，实验策略带来了真实且显著的负向影响。")
    else:
        print(f" ⚠️ 结果在统计上不显著 (p >= {alpha})。")
        print("    我们没有足够的证据表明两组之间存在真实差异，观测到的差异可能是由随机波动引起的。")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="A/B 实验评估工具")
    parser.add_argument("control_path", help="对照组CSV文件的路径")
    parser.add_argument("treatment_path", help="实验组CSV文件的路径")
    args = parser.parse_args()
    analyze_ab_test(args.control_path, args.treatment_path)

if __name__ == "__main__":
    main()