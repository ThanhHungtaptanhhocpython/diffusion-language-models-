#!/usr/bin/env python3
"""
Khung kiem thu so hoc doc lap (Zero-dependency Numerical Verification Harness)
cho chuyen khao Diffusion Language Models (AIO2026).
Tac gia: Pham Thanh Hung.

Kiem thu tat ca 5 phep tinh tay trong cac Phan II.1, III.1, IV.3, VI.2 va VII.2.
Chay bang Python thuan (chi dung thu vien chuan math, khong can torch hoac numpy).
Moi phep tinh duoc xac thuc chinh xac den 4 chu so thap phan, khop voi van ban LaTeX.
Ma thoat: 0 khi tat ca cac khang dinh (assertions) deu vuot qua 100%.
"""

import math
import sys

def matmul_4x4(A, B):
    """Nhan hai ma tran 4x4 danh cho danh sach 2 chieu thuan tuy."""
    C = [[0.0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C

def verify_section_02_autoregressive():
    """
    Kiem thu Phep tinh tay Phan II.1:
    Phan ra xac suat chuoi tu hoi quy (Autoregressive 3-token factorization).
    Chuoi 3 token: p(x1)=0.4000, p(x2|x1)=0.2500, p(x3|x1:2)=0.8000.
    """
    print("=" * 65)
    print("[1/5] KIEM THU PHAN II.1: MO HINH TU HOI QUY (AUTOREGRESSIVE)")
    print("=" * 65)
    p1 = 0.4000
    p2 = 0.2500
    p3 = 0.8000

    # 1. Tich xac suat dong thoi
    p_joint = p1 * p2 * p3
    assert abs(p_joint - 0.0800) < 1e-6, f"Sai lech p_joint: {p_joint}"

    # 2. Cac so hang log-likelihood tu nhien (ln)
    log_p1 = math.log(p1)
    log_p2 = math.log(p2)
    log_p3 = math.log(p3)
    sum_log_p = log_p1 + log_p2 + log_p3

    assert abs(log_p1 - (-0.9163)) < 1e-4, f"Sai lech log_p1: {log_p1}"
    assert abs(log_p2 - (-1.3863)) < 1e-4, f"Sai lech log_p2: {log_p2}"
    assert abs(log_p3 - (-0.2231)) < 1e-4, f"Sai lech log_p3: {log_p3}"
    assert abs(sum_log_p - (-2.5257)) < 1e-4, f"Sai lech sum_log_p: {sum_log_p}"

    # 3. Ham mat mat NLL va do roi loan Perplexity
    nll_loss = -sum_log_p / 3.0
    ppl = math.exp(nll_loss)

    assert abs(nll_loss - 0.8419) < 1e-4, f"Sai lech NLL loss: {nll_loss}"
    assert abs(ppl - 2.3208) < 1e-4, f"Sai lech Perplexity: {ppl}"

    print(f"  + Xac suat dong thoi p(x) = {p_joint:.4f}")
    print(f"  + Tong log-likelihood ln p(x) = {sum_log_p:.4f}")
    print(f"  + Mat mat NLL tren moi token = {nll_loss:.4f}")
    print(f"  + Do roi loan Perplexity (PPL) = {ppl:.4f}")
    print("  => KET QUA PHAN II.1: DAT (PASS)\n")


def verify_section_03_d3pm_transition():
    """
    Kiem thu Phep tinh tay Phan III.1:
    Ma tran chuyen trang thai va phan phoi hau nghiem D3PM (Austin et al., NeurIPS 2021).
    Khong gian trang thai K=4: [A, B, C, [M]].
    """
    print("=" * 65)
    print("[2/5] KIEM THU PHAN III.1: MA TRAN CHUYEN TRANG THAI D3PM")
    print("=" * 65)
    beta1 = 0.2000
    beta2 = 0.3000

    alpha1 = 1.0 - beta1         # 0.8000
    alpha2 = 1.0 - beta2         # 0.7000
    alpha_bar_2 = alpha1 * alpha2 # 0.5600
    mask_prob_2 = 1.0 - alpha_bar_2 # 0.4400

    assert abs(alpha1 - 0.8000) < 1e-6
    assert abs(alpha_bar_2 - 0.5600) < 1e-6
    assert abs(mask_prob_2 - 0.4400) < 1e-6

    # Ma tran Q1 (4x4)
    Q1 = [
        [1.0 - beta1, 0.0, 0.0, beta1],
        [0.0, 1.0 - beta1, 0.0, beta1],
        [0.0, 0.0, 1.0 - beta1, beta1],
        [0.0, 0.0, 0.0, 1.0]
    ]

    # Ma tran Q2 (4x4)
    Q2 = [
        [1.0 - beta2, 0.0, 0.0, beta2],
        [0.0, 1.0 - beta2, 0.0, beta2],
        [0.0, 0.0, 1.0 - beta2, beta2],
        [0.0, 0.0, 0.0, 1.0]
    ]

    Q_bar_2 = matmul_4x4(Q1, Q2)

    # Kiem tra cac phan tu cua hang dau tien (trang thai ban dau x0 = A)
    assert abs(Q_bar_2[0][0] - 0.5600) < 1e-6, f"Q_bar_2[0][0]: {Q_bar_2[0][0]}"
    assert abs(Q_bar_2[0][1] - 0.0000) < 1e-6
    assert abs(Q_bar_2[0][2] - 0.0000) < 1e-6
    assert abs(Q_bar_2[0][3] - 0.4400) < 1e-6, f"Q_bar_2[0][3]: {Q_bar_2[0][3]}"
    assert abs(Q_bar_2[3][3] - 1.0000) < 1e-6

    # Tinh phan phoi hau nghiem: q(x1 | x2=[M], x0=A)
    # q(x1=A | x2=[M], x0=A) = Q2[A, [M]] * Q1[A, A] / Q_bar_2[A, [M]] = 0.30 * 0.80 / 0.44 = 0.24 / 0.44 = 6/11
    # q(x1=[M] | x2=[M], x0=A) = Q2[[M], [M]] * Q1[A, [M]] / Q_bar_2[A, [M]] = 1.00 * 0.20 / 0.44 = 0.20 / 0.44 = 5/11
    post_A = (0.30 * 0.80) / 0.44
    post_M = (1.00 * 0.20) / 0.44

    assert abs(post_A - 0.5455) < 1e-4, f"Sai lech post_A: {post_A}"
    assert abs(post_M - 0.4545) < 1e-4, f"Sai lech post_M: {post_M}"
    assert abs((post_A + post_M) - 1.0000) < 1e-6

    print(f"  + He so giu lai tich luy alpha_bar_2 = {alpha_bar_2:.4f}")
    print(f"  + Xac suat bi mat na o buoc 2 = {mask_prob_2:.4f}")
    print(f"  + Hang dau tien cua Q_bar_2: {[round(x, 4) for x in Q_bar_2[0]]}")
    print(f"  + Hau nghiem q(x1=A | x2=[M], x0=A) = 0.24 / 0.44 = {post_A:.4f} (~ 6/11)")
    print(f"  + Hau nghiem q(x1=[M] | x2=[M], x0=A) = 0.20 / 0.44 = {post_M:.4f} (~ 5/11)")
    print("  => KET QUA PHAN III.1: DAT (PASS)\n")


def verify_section_04_mdlm_loss():
    """
    Kiem thu Phep tinh tay Phan IV.3 / V.2:
    Ham mat mat Cross-Entropy co trong so 1/t cua Masked Diffusion (Sahoo et al., 2024; Shi et al., 2024).
    Cau 5 token 'con meo ngoi tren tham', t=0.4000, mat na 2 vi tri (meo va tren).
    """
    print("=" * 65)
    print("[3/5] KIEM THU PHAN IV.3: HAM MAT MAT MASKED DIFFUSION (MDLM / MD4)")
    print("=" * 65)
    t = 0.4000
    weight = 1.0 / t  # 2.5000
    assert abs(weight - 2.5000) < 1e-6

    # Truong hop 1: Tinh tu logits qua Softmax day du (nhu handoff explorer)
    # Vi tri 2: logits = [2.1, 0.5, -0.2, -0.8, -1.0] -> target 'meo' o chi so 0
    logits2 = [2.1000, 0.5000, -0.2000, -0.8000, -1.0000]
    exp2 = [math.exp(v) for v in logits2]
    sum_exp2 = sum(exp2)
    p2_target = exp2[0] / sum_exp2
    ce2 = -math.log(p2_target)

    assert abs(sum_exp2 - 11.4509) < 1e-3, f"Sai lech sum_exp2: {sum_exp2}"
    assert abs(p2_target - 0.7132) < 1e-4, f"Sai lech p2_target: {p2_target}"
    assert abs(ce2 - 0.3380) < 1e-4, f"Sai lech ce2: {ce2}"

    # Vi tri 4: logits = [1.8, 1.1, 0.2, -0.5, -1.2] -> target 'tren' o chi so 0
    logits4 = [1.8000, 1.1000, 0.2000, -0.5000, -1.2000]
    exp4 = [math.exp(v) for v in logits4]
    sum_exp4 = sum(exp4)
    p4_target = exp4[0] / sum_exp4
    ce4 = -math.log(p4_target)

    assert abs(sum_exp4 - 11.1829) < 1e-3, f"Sai lech sum_exp4: {sum_exp4}"
    assert abs(p4_target - 0.5410) < 1e-4, f"Sai lech p4_target: {p4_target}"
    assert abs(ce4 - 0.6144) < 1e-4, f"Sai lech ce4: {ce4}"

    sum_ce = ce2 + ce4
    avg_ce = sum_ce / 2.0
    weighted_loss = weight * sum_ce

    assert abs(sum_ce - 0.9524) < 1e-4, f"Sai lech sum_ce: {sum_ce}"
    assert abs(avg_ce - 0.4762) < 1e-4, f"Sai lech avg_ce: {avg_ce}"
    assert abs(weighted_loss - 2.3811) < 1e-3, f"Sai lech weighted_loss: {weighted_loss}"

    # Truong hop 2: Truong hop don gian hoa voi xac suat tron p_meo=0.70, p_tren=0.50 (nhu literature explorer)
    ce_simple_meo = -math.log(0.7000)   # 0.3567
    ce_simple_tren = -math.log(0.5000)  # 0.6931
    sum_ce_simple = ce_simple_meo + ce_simple_tren # 1.0498
    weighted_simple = weight * sum_ce_simple       # 2.6246
    assert abs(ce_simple_meo - 0.3567) < 1e-4
    assert abs(ce_simple_tren - 0.6931) < 1e-4
    assert abs(weighted_simple - 2.6246) < 1e-4

    print(f"  + Trong so thoi gian w(t) = 1/t = 1/0.4000 = {weight:.4f}")
    print(f"  + Vi tri 2 ('meo'): p_theta = {p2_target:.4f}, CE = -ln(p) = {ce2:.4f}")
    print(f"  + Vi tri 4 ('tren'): p_theta = {p4_target:.4f}, CE = -ln(p) = {ce4:.4f}")
    print(f"  + Tong mat mat CE tren cac vi tri bi che = {sum_ce:.4f}")
    print(f"  + Mat mat co trong so L_MDLM = w(t) * sum(CE) = {weighted_loss:.4f}")
    print("  => KET QUA PHAN IV.3: DAT (PASS)\n")


def verify_section_06_sampling_remasking():
    """
    Kiem thu Phep tinh tay Phan VI.2:
    Thuat toan lay mau gỡ mat na dua tren do tu tin (Confidence-based Remasking - LLaDA).
    """
    print("=" * 65)
    print("[4/5] KIEM THU PHAN VI.2: REMASKING DUA TREN DO TU TIN")
    print("=" * 65)
    confidences = {
        "pos1_troi": 0.8500,
        "pos2_hom":  0.4200,
        "pos3_nay":  0.6800,
        "pos4_mua":  0.9100
    }
    sorted_conf = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    top2_positions = [item[0] for item in sorted_conf[:2]]
    remasked_positions = [item[0] for item in sorted_conf[2:]]

    assert top2_positions == ["pos4_mua", "pos1_troi"], f"Sai lech top 2: {top2_positions}"
    assert remasked_positions == ["pos3_nay", "pos2_hom"], f"Sai lech remasked: {remasked_positions}"

    print(f"  + Do tu tin ban dau: {confidences}")
    print(f"  + Thu tu xep hang giam dan: {sorted_conf}")
    print(f"  + Cac vi tri duoc go mat na buoc 1 (top 2): {top2_positions}")
    print(f"  + Cac vi tri bi che lai ([M]) o buoc 1: {remasked_positions}")
    print("  => KET QUA PHAN VI.2: DAT (PASS)\n")


def verify_section_07_parallel_independence_and_kv_cache():
    """
    Kiem thu Phep tinh tay Phan VII.2 & VI.5:
    1. Loi gia dinh doc lap khi giai ma song song 1 buoc ('New York' vs 'San Francisco').
    2. Tiet kiem chi phi tinh toan cua Block Diffusion voi KV Cache.
    """
    print("=" * 65)
    print("[5/5] KIEM THU PHAN VII.2: DOC LAP SONG SONG VA TIET KIEM KV-CACHE")
    print("=" * 65)
    # Phan 1: Loi chimeric khi gia dinh doc lap
    p_NY = 0.5000
    p_SF = 0.5000

    # Xac suat bien tai tung vi tri
    p_pos1_New = 0.5000
    p_pos1_San = 0.5000
    p_pos2_York = 0.5000
    p_pos2_Fran = 0.5000

    # Tich xac suat phan ra doc lap q(x1, x2) = p(x1) * p(x2)
    q_NY = p_pos1_New * p_pos2_York  # 0.2500
    q_SF = p_pos1_San * p_pos2_Fran  # 0.2500
    q_NF = p_pos1_New * p_pos2_Fran  # 0.2500 (loi lai ghep chimeric)
    q_SY = p_pos1_San * p_pos2_York  # 0.2500 (loi lai ghep chimeric)

    assert abs(q_NY - 0.2500) < 1e-6
    assert abs(q_SF - 0.2500) < 1e-6
    assert abs(q_NF - 0.2500) < 1e-6
    assert abs(q_SY - 0.2500) < 1e-6

    total_chimeric = q_NF + q_SY
    assert abs(total_chimeric - 0.5000) < 1e-6

    # Khoang cach bien thien toan phan (Total Variation Distance)
    # P_true: NY=0.5, SF=0.5, NF=0.0, SY=0.0
    d_tv = 0.5 * (abs(0.50 - q_NY) + abs(0.50 - q_SF) + abs(0.00 - q_NF) + abs(0.00 - q_SY))
    assert abs(d_tv - 0.5000) < 1e-6

    print("  --- 1. Danh gia loi Doc lap Song song ---")
    print(f"  + Xac suat sinh dung: q(New York) = {q_NY:.4f}, q(San Francisco) = {q_SF:.4f}")
    print(f"  + Xac suat loi lai ghep: q(New Francisco) = {q_NF:.4f}, q(San York) = {q_SY:.4f}")
    print(f"  + Tong ty le loi lai ghep chimeric = {total_chimeric * 100:.2f}%")
    print(f"  + Khoang cach bien thien toan phan D_TV = {d_tv:.4f}")

    # Phan 2: Tiet kiem chi phi phep tinh Attention trong Block Diffusion
    L = 512       # Do dai chuoi tong cong
    B = 32        # Kich thuoc moi khoi (block)
    M = L // B    # 16 khoi
    S_full = 32   # So buoc khuet tan DLM toan phan khong co KV Cache
    S_b = 4       # So buoc khuet tan trong moi block voi Block Diffusion

    # DLM toan phan: S_full * L^2
    ops_full = S_full * (L ** 2) # 32 * 262,144 = 8,388,608

    # Block Diffusion co luu tru KV Cache cac khoi qua khu:
    # Voi block m (0 <= m < M): Query = B, Key = (m + 1) * B
    ops_bd = sum(S_b * (B ** 2) * (m + 1) for m in range(M)) # 4 * 1024 * (16*17/2) = 557,056

    assert ops_full == 8388608
    assert ops_bd == 557056

    speedup = ops_full / ops_bd
    reduction_pct = (1.0 - ops_bd / ops_full) * 100.0

    assert abs(speedup - 15.0588) < 1e-3
    assert abs(reduction_pct - 93.3594) < 1e-3

    print("\n  --- 2. Tiet kiem phep tinh Block Diffusion & KV Cache ---")
    print(f"  + DLM toan phan (L={L}, S={S_full}): {ops_full:,} phep tinh token-pair")
    print(f"  + Block Diffusion (B={B}, S_b={S_b}, M={M}): {ops_bd:,} phep tinh token-pair")
    print(f"  + Ty le tang toc ly thuyet: {speedup:.2f}x (giam {reduction_pct:.2f}% chi phi)")
    print("  => KET QUA PHAN VII.2: DAT (PASS)\n")


def main():
    print("\n" + "#" * 65)
    print("#  KHUNG KIEM THU SO HOC TOAN BO CHO CHUYEN KHAO DLM (AIO2026)   #")
    print("#" * 65 + "\n")

    verify_section_02_autoregressive()
    verify_section_03_d3pm_transition()
    verify_section_04_mdlm_loss()
    verify_section_06_sampling_remasking()
    verify_section_07_parallel_independence_and_kv_cache()

    print("=" * 65)
    print("XAC THUC HOAN TAT: 100% PHEP TINH TAY DUOC VERIFIED CHINH XAC!")
    print("TAT CA ASSERTIONS HOAN TOAN KHOP VOI VAN BAN VA CONG THUC.")
    print("=" * 65 + "\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
