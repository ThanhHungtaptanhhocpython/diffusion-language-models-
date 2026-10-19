# Diffusion Language Models: Từ Khử Nhiễu Rời Rạc Đến Mô Hình Ngôn Ngữ Lớn

[![AI VIET NAM](https://img.shields.io/badge/Course-AIO2026-brightgreen.svg)](https://aivietnam.edu.vn)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhHungtaptanhhocpython/diffusion-language-models-/blob/main/case_study/mini_dlm_case_study.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PDF Monograph](https://img.shields.io/badge/PDF-28%20Pages-red.svg)](main.pdf)

> **Tác giả:** Phạm Thanh Hưng  
> **Chương trình:** All-in-One Course 2026 (AIO2026) — AI VIET NAM  
> **Định dạng:** Chuyên khảo học thuật (28 trang PDF) kèm mã nguồn thực nghiệm và bộ công cụ kiểm toán độc lập.

---

## 📌 Giới Thiệu Chuyên Đề

Trong hơn nửa thập kỷ qua, trí tuệ nhân tạo tạo sinh (*Generative AI*) được dẫn dắt bởi các mô hình ngôn ngữ lớn tự hồi quy (*Autoregressive Large Language Models* – AR LLMs). Tuy nhiên, cơ chế sinh tuần tự từ trái sang phải áp đặt 3 điểm nghẽn cố hữu:
1. **Độ trễ suy luận $\mathcal{O}(L)$** bị nghẽn băng thông bộ nhớ (*Memory-Bound*).
2. **Bất lực trong tác vụ điền vào giữa (*Infilling*)** và chỉnh sửa cục bộ do thiếu ngữ cảnh tương lai.
3. **Lời nguyền đảo ngược (*The Reversal Curse*)**: mô hình học xuôi $A \to B$ nhưng thất bại khi suy luận ngược $B \to A$.

Kho lưu trữ này cung cấp toàn bộ tài liệu nghiên cứu chuyên sâu, mã nguồn huấn luyện mô hình **Mini-DLM** (*Discrete Masked Diffusion Language Model*) và script kiểm chứng số học độc lập cho mô thức khuếch tán ngôn ngữ.

---

## 📂 Cấu Trúc Kho Lưu Trữ

```text
.
├── README.md                               # Hướng dẫn chi tiết & liên kết học tập
├── main.pdf                                # Chuyên khảo hoàn chỉnh (28 trang PDF)
├── case_study/                             # Mã nguồn thực hành cho học viên
│   ├── mini_dlm_case_study.ipynb           # Sổ tay Jupyter Notebook (chạy CPU / Colab T4)
│   └── mini_dlm_case_study.py              # Script Python tương đương
└── scripts/                                # Công cụ kiểm toán toán học
    └── verify_hand_calculations.py         # Kiểm toán độc lập 5 bài toán tính tay
```

---

## 🚀 Hướng Dẫn Thực Hành Nhanh

### Cách 1: Chạy trực tiếp trên Google Colab (Khuyên dùng)
Học viên chỉ cần nhấp vào huy hiệu bên dưới để mở sổ tay và chạy thử nghiệm mà không cần cài đặt môi trường:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhHungtaptanhhocpython/diffusion-language-models-/blob/main/case_study/mini_dlm_case_study.ipynb)

*Thời gian huấn luyện trên Google Colab GPU T4: $\approx 10$ giây.*

---

### Cách 2: Chạy cục bộ trên máy tính cá nhân

1. **Clone kho mã nguồn về máy:**
   ```bash
   git clone https://github.com/ThanhHungtaptanhhocpython/diffusion-language-models-.git
   cd diffusion-language-models-
   ```

2. **Cài đặt thư viện phụ thuộc (Python 3.10+):**
   ```bash
   pip install torch numpy jupyter
   ```

3. **Chạy Jupyter Notebook:**
   ```bash
   jupyter notebook case_study/mini_dlm_case_study.ipynb
   ```
   *Hoặc chạy script Python trực tiếp trong terminal:*
   ```bash
   python3 case_study/mini_dlm_case_study.py
   ```

---

### Cách 3: Kiểm chứng các phép tính tay lý thuyết

Kho lưu trữ cung cấp script kiểm toán độc lập bằng Python thuần (không phụ thuộc vào PyTorch hay NumPy) nhằm xác thực 100% các giá trị trong bảng tính tay của bài đọc:

```bash
python3 scripts/verify_hand_calculations.py
```

**Kết quả kiểm toán bao gồm:**
* Xác thực phân rã chuỗi tự hồi quy AR, NLL Loss và Perplexity (Bảng 1).
* Xác thực ma trận tích lũy $\bar{Q}_2 = Q_1 Q_2$ và phân phối hậu nghiệm Bayes $q(x_1 \mid x_2=\texttt{[M]}, x_0=A)$ của D3PM (Bảng 3).
* Xác thực hàm mất mát Cross-Entropy có trọng số thời gian $1/t$ của MDLM/MD4 (Bảng 4).
* Xác thực thuật toán lấy mẫu Confidence-based Remasking của LLaDA (Bảng 8).

---

## 📊 Ma Trận So Sánh Nhanh: AR LLM vs DLM

| Tiêu chí | Mô hình Tự hồi quy (AR LLM) | Mô hình Ngôn ngữ Khuếch tán (DLM) |
| :--- | :--- | :--- |
| **Cơ chế Attention** | Causal Mask (nhân quả một chiều) | Full Bidirectional Attention (hai chiều toàn cục) |
| **Độ trễ suy luận** | $\mathcal{O}(L)$ tuyến tính theo độ dài | $\mathcal{O}(S)$ độc lập với độ dài chuỗi ($S \ll L$) |
| **Bản chất tính toán** | Memory-bound (nghẽn băng thông RAM/VRAM) | Compute-bound (tận dụng tối đa Tensor Core GPU) |
| **Tác vụ Infilling** | Kém hoặc phải huấn luyện cấu trúc đặc biệt | Tự nhiên, chỉ cần ghim token bối cảnh và unmask |
| **Lời nguyền đảo ngược** | Mắc phải ($A \to B \not\Rightarrow B \to A$) | Khắc phục triệt để nhờ cơ chế khử nhiễu toàn thể |
| **KV Cache** | Hoạt động hoàn hảo | Thách thức lớn (đòi hỏi Approximate KV Cache / BD3-LM) |

---

## 📖 Trích Dẫn & Bản Quyền

Tài liệu được phát triển phục vụ công tác đào tạo và nghiên cứu học thuật tại **AI VIET NAM (AIO2026)**.

```bibtex
@misc{pham2026dlm,
  author    = {Pham, Thanh Hung},
  title     = {Diffusion Language Models: Tu Khu Nhieu Roi Rac Den Mo Hinh Ngon Ngu Lon},
  year      = {2026},
  publisher = {AI VIET NAM},
  series    = {All-in-One Course 2026 (AIO2026)},
  url       = {https://github.com/ThanhHungtaptanhhocpython/diffusion-language-models-}
}
```

