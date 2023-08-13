import math

class lagrange(object):
    def calculate_fr(f_r_n, a, m):
        denominator = sum(math.sqrt(a[i][m] * C[i]) for i in range(len(a)) if i != m)
        f_r_star_mn = f_r_n / denominator
        return f_r_star_mn

    def calculate_floc(f_loc, b, m):
        denominator = sum(math.sqrt(b[i] * C[i]) for i in range(len(b)) if i != m)
        f_loc_star_m = f_loc / denominator
        return f_loc_star_m

    # 示例参数
    f_r_n = 10
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    f_loc = 20
    b = [1, 2, 3]
    C = [0.5, 0.7, 0.9]

    # 计算 f^r*_{mn}
    m = 1
    f_r_star_mn = calculate_fr(f_r_n, a, m)
    print( {f_r_star_mn})

    # 计算 f^{loc*}_m
    m = 2
    f_loc_star_m = calculate_floc(f_loc, b, m)
    print({f_loc_star_m})
