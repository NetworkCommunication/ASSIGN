class adapt(object):
    def calculate_beta_e(sigma, f_n_max, p_r, f_n_max_total, varrho, f_loc_avg, f_off_avg, varpi):
        numerator = sum(f_n_max[:p_r])
        denominator = sum(f_n_max_total)
        term1 = sigma * (1 - numerator / denominator)
        term2 = varrho * f_loc_avg / f_off_avg
        beta_e = varpi * (term1 + term2)
        return beta_e

    def calculate_varpi(S, S_star, N):
        numerator = sum(1 - (S - S_star) / S for _ in range(N))
        varpi = numerator / N
        return varpi

    def calculate_beta_t(beta_e):
        beta_t = 1 - beta_e
        return beta_t

    # 示例参数
    sigma = 0.5
    varpi = 0.5
    f_n_max = [10, 20, 30, 40]
    p_r = 3
    f_n_max_total = [50, 60, 70, 80, 90]
    varrho = 0.2
    f_loc_avg = 25
    f_off_avg = 35
    S = 100
    S_star = 80
    N = 4

    # 计算 beta_e
    beta_e = calculate_beta_e(sigma, f_n_max, p_r, f_n_max_total, varrho, f_loc_avg, f_off_avg)
    print(beta_e)

    # 计算 varpi
    varpi = calculate_varpi(S, S_star, N)
    print(varpi)

    # 计算 beta_t
    beta_t = calculate_beta_t(beta_e)
    print(beta_t)

