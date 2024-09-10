import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def binomial_test(sample_size, defective_items, defective_rate, confidence_level):
    """
    使用二项分布进行假设检验，并绘制正态分布图

    :param sample_size: 样本大小
    :param defective_items: 次品数量
    :param defective_rate: 标定次品率
    :param confidence_level: 置信区间
    :return: 是否接受零配件
    """
    # 计算假设次品率下的显著性水平
    alpha = 1 - confidence_level

    # 使用 binomial_test 进行检验，计算拒绝零假设的 p 值
    result = stats.binomtest(defective_items, sample_size, defective_rate, alternative='greater')

    # 计算二项分布的均值和标准差
    mu = sample_size * defective_rate
    sigma = np.sqrt(sample_size * defective_rate * (1 - defective_rate))

    # 设置绘图范围
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

    # 计算正态分布的 概率密度函数
    normal_pdf = stats.norm.pdf(x, mu, sigma)

    # 绘制正态分布图
    plt.figure(figsize=(10, 6))
    plt.plot(x, normal_pdf, label='Normal Approximation', color='blue')
    plt.fill_between(x, normal_pdf, alpha = 1 - confidence_level, color='blue')

    # 绘制二项分布的概率质量函数
    binom_pmf = stats.binom.pmf(range(sample_size + 1), sample_size, defective_rate)
    plt.stem(range(sample_size + 1), binom_pmf, basefmt='C1-', linefmt='C1-', markerfmt='C1o', label='Binomial PMF')

    # 标记样本中的次品数量
    plt.axvline(defective_items, color='green', linestyle='--', label=f'Defective Items: {defective_items}')

    plt.title('Binomial Distribution and Normal Approximation')
    plt.xlabel('Number of Defective Items')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 如果 p 值小于显著性水平就拒绝原假设
    if result.pvalue < alpha:
        return f'拒收零配件(p-value:{result.pvalue:.5f})'
    else:
        return f'接收零配件(p-value:{result.pvalue:.5f})'


def calculate_min_sample_size(defective_rate, confidence_level, error_margin):
    """
    计算最小样本量
    :param defective_rate:标称次品率
    :param confidence_level: 置信区间
    :param error_margin: 允许的误差范围
    :return: 最小样本量
    """
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    print('z_score=', z_score)
    sample_size = (z_score ** 2 * defective_rate * (1 - defective_rate) / (error_margin ** 2))
    return int(np.ceil(sample_size))


def detection_cost(n, c):
    return n * c


# 设定各项初始值：
defective_rate = 0.10
confidence_level_95 = 0.95
confidence_level_90 = 0.90
error_margin = 0.05

sample_size_95 = calculate_min_sample_size(defective_rate, confidence_level_95, error_margin)
sample_size_90 = calculate_min_sample_size(defective_rate, confidence_level_90, error_margin)

print(f'95% 置信水平下的最小样本量：{sample_size_95}')
print(f'90% 置信水平下的最小样本量：{sample_size_90}')

detected_defective_items = 10  # 假设我们抽到的样本中有10个次品
cost1 = detection_cost(sample_size_90, 2.583)  # 这里取图一零配件检测成本的均值=2.583
cost2 = detection_cost(sample_size_95, 2.583)
# 对两种置信水平进行二项分布假设检验
result_95 = binomial_test(sample_size_95, detected_defective_items, defective_rate, confidence_level_95)
result_90 = binomial_test(sample_size_90, detected_defective_items, defective_rate, confidence_level_90)
print('90%信度下的最低成本为：', cost1)
print('95%信度下的最低成本为：', cost2)
