# 📈 Hệ Thống Kiểm Định Định Lượng HOSE V4: Nested Walk-Forward Portfolio Management

Ứng dụng Web App tương tác chuyên sâu xây dựng trên nền tảng **Streamlit**, phục vụ việc backtest (kiểm định lịch sử), phân tích danh mục định lượng và tối ưu hóa phân bổ tỷ trọng theo phương pháp **Nested Walk-Forward Optimization (FA/TA → Top N → EW / MPT / MinVar / HRP)** trên toàn bộ dữ liệu 5 năm (2021 – 2025) của 97 cổ phiếu hàng đầu sàn HOSE.

Dự án được xây dựng tuân thủ các tiêu chuẩn định lượng khắt khe nhất trong giới quỹ đầu tư: **Triệt tiêu hoàn toàn Future Leakage (Look-ahead Bias)**, **Lựa chọn danh mục theo dữ liệu As-Of phiên $t-1$**, **Hạch toán chi phí giao dịch thực tế (Commission + Slippage)**, và **Cơ chế quản trị rủi ro đa tầng (Market Regime EMA200 & Cash Reserve)**.

---

## 🌟 Tính Năng Nổi Bật Của Web App

Ứng dụng được thiết kế theo chuẩn giao diện Fintech cao cấp với **8 Tab chức năng chuyên biệt**:

1. **📊 1. Dữ Liệu & Phân Khúc (Data Quality & Splits):**
   - Tự động nhận diện file dữ liệu thị trường trong thư mục hoặc qua hộp tải file CSV / Excel.
   - Bảng kiểm tra chất lượng dữ liệu từng mã: Số phiên giao dịch, tỷ lệ missing volume, thanh khoản trung vị.
   - Trực quan hóa phân chia 3 tập thời gian: **Core Train (2021–2023)**, **Validation (2024)**, và **Final Test (2025)**.

2. **⚙️ 2. Tối Ưu Ngưỡng Core Train (Threshold Optimization):**
   - Khóa cặp ngưỡng `BUY` (5, 6, 7) và `EXIT` (2, 3, 4) tối ưu dựa trên Cross-Validation với hàm phạt mất ổn định (Stability Penalty).
   - Biểu đồ nhiệt so sánh điểm số ổn định Global CV Score giữa các cặp ngưỡng.

3. **🔄 3. Lịch Trình Top N (Nested Walk-Forward Selection):**
   - Danh mục Top N được tái lựa chọn theo quý một cách linh hoạt bằng thông tin **chỉ đến phiên $t-1$**.
   - Bảng xếp hạng chi tiết các cổ phiếu qua 2 vòng lọc rủi ro (**Pass TA Gate** và **Pass Risk Gate**) kết hợp **Correlation Cap** ($|\rho| \le 0.80$) nhằm phân tán rủi ro.

4. **🏆 4. Đấu Trường Validation (Model Competition & Champion Selection):**
   - Cuộc đua sòng phẳng giữa 5 họ mô hình phân bổ danh mục:
     * **Dynamic EW**: Phân bổ đều Equal Weight.
     * **Dynamic MPT-LW**: Tối đa hóa Sharpe với Covariance Shrinkage (Ledoit–Wolf) và L2 Regularization.
     * **Dynamic MinVar-LW**: Tối thiểu hóa phương sai danh mục với Ledoit–Wolf.
     * **Dynamic HRP**: Phân bổ theo cấu trúc phân cấp rủi ro (Hierarchical Risk Parity).
     * **Dynamic Blend50**: Kết hợp 50% MPT-LW + 50% Equal Weight.
   - Tự động chấm điểm `ValidationScore` và phong tặng danh hiệu **LOCKED CHAMPION**.

5. **🎯 5. Kiểm Định Final Test (Out-of-Sample Performance):**
   - Đánh giá độc lập mô hình Champion trên tập Out-of-Sample chưa từng thấy.
   - Thẻ KPI đo lường toàn diện: *Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Excess CAGR vs Benchmark, Max Drawdown Improvement*.
   - Biểu đồ tương tác Plotly: **Đường tăng trưởng tài sản (NAV Curves)** và **Đồ thị sụt giảm vốn (Underwater Drawdown Chart)**.

6. **📋 6. Nhật Ký Lệnh & Trọng Số (Trade Log & Weights):**
   - Xuất chi tiết từng lệnh khớp giá Open: *Ngày, Mã, Chiều Mua/Bán, Số lượng, Giá khớp, Phí giao dịch*.
   - Hỗ trợ **tải file CSV Nhật ký giao dịch** chỉ với 1 click.
   - Đồ thị diện tích trực quan hóa tỷ lệ phân bổ Cổ phiếu vs Tiền mặt (Cash Reserve) theo thời gian.

7. **🔍 7. Phân Tích Mã Riêng Lẻ (Single Stock Deep-Dive):**
   - Khám phá kỹ thuật bất kỳ mã nào trong 97 mã cổ phiếu.
   - Biểu đồ nến Candlestick kết hợp EMA 20, 50, 200, dải khối lượng và dòng tiền OBV.
   - Đồ thị lịch sử điểm số kỹ thuật **TA Score (0–10)** và trạng thái vị thế (Mua / Tiền mặt).

8. **🛡️ 8. Audit Chống Leakage (Leakage Audit & Methodology):**
   - Ma trận 7 bước kiểm soát rò rỉ thông tin tương lai nghiêm ngặt.
   - Trình bày công thức toán học và phương pháp luận định lượng nền tảng.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
TAO-APP-CHIEN-LUOC/
│
├── app.py                                              # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt                                    # Danh sách thư viện Python phụ thuộc
├── README.md                                           # Hướng dẫn sử dụng và triển khai dự án
├── HOSE - 01.01.2021 - 31.12.2025 (97 mã).csv          # Dữ liệu OHLCV 5 năm của 97 cổ phiếu HOSE
└── HOSE_V4_GENERIC_NESTED_WALK_FORWARD_STEP_BY_STEP_VI.ipynb # Notebook nghiên cứu gốc
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Máy tính cài đặt **Python 3.9 trở lên** (khuyên dùng Python 3.10 hoặc 3.11).
- Trình duyệt web hiện đại (Google Chrome, MS Edge, Safari, Firefox).

### 2. Các bước khởi chạy
Mở ứng dụng dòng lệnh (**Terminal** trên macOS/Linux hoặc **PowerShell / Command Prompt** trên Windows):

```bash
# Bước 1: Di chuyển vào thư mục dự án
cd "đường-dẫn-đến-thư-mục-dự-án"

# Bước 2: Tạo và kích hoạt môi trường ảo (Khuyên dùng)
# Trên macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Trên Windows:
python -m venv venv
venv\Scripts\activate

# Bước 3: Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Bước 4: Khởi chạy ứng dụng Streamlit
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Đẩy Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 1: Đẩy Mã Nguồn Lên GitHub

1. Truy cập [GitHub.com](https://github.com) và đăng nhập tài khoản.
2. Tạo một Repository mới (ví dụ đặt tên: `hose-v4-quant-strategy`), chọn quyền **Public**.
3. Tại Terminal trong thư mục dự án trên máy tính, thực hiện các lệnh sau:

```bash
# 1. Khởi tạo Git repository
git init

# 2. Thêm các file cần thiết vào Git
git add app.py requirements.txt README.md "HOSE - 01.01.2021 - 31.12.2025 (97 mã).csv"

# 3. Tạo commit đầu tiên
git commit -m "feat: Khoi tao ung dung Streamlit dinh luong HOSE V4"

# 4. Đổi tên nhánh chính thành main
git branch -M main

# 5. Liên kết tới remote repository trên GitHub (thay URL của bạn vào)
git remote add origin https://github.com/TÊN_GITHUB_CỦA_BẠN/hose-v4-quant-strategy.git

# 6. Đẩy code lên GitHub
git push -u origin main
```

*(Lưu ý: Nếu file CSV lớn hơn 100MB, bạn có thể nén zip hoặc để người dùng tự upload file trực tiếp trên giao diện Web App).*

---

### Bước 2: Deploy Miễn Phí Trên Streamlit Community Cloud

1. Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Điền thông tin cấu hình:
   - **Repository:** Chọn repository bạn vừa tạo (`TÊN_GITHUB_CỦA_BẠN/hose-v4-quant-strategy`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**.
5. Trong vòng 1–2 phút, Streamlit Cloud sẽ tự động cài đặt các gói trong `requirements.txt` và khởi chạy Web App của bạn với một đường link công khai dạng `https://ten-app.streamlit.app` để chia sẻ cho mọi người cùng sử dụng!

---

## 💡 Xử Lý Sự Cố Thường Gặp (Troubleshooting)

- **Lỗi thiếu file dữ liệu khi deploy:** Ứng dụng đã tích hợp sẵn công cụ tải file trực tiếp trên thanh Sidebar. Bạn chỉ cần chọn *"Tải lên file CSV / Excel mới"* và kéo thả file vào là ứng dụng hoạt động ngay.
- **Lỗi font chữ tiếng Việt trên biểu đồ:** Các biểu đồ Plotly trong ứng dụng sử dụng chuẩn UTF-8 trực tiếp từ trình duyệt, đảm bảo hiển thị tiếng Việt hoàn hảo trên cả Windows, macOS và điện thoại di động.

---

## 👨‍💻 Tác Quyền & Đóng Góp

- **Chiến lược:** Mô hình định lượng HOSE V4 Nested Walk-Forward.
- **Nền tảng:** Python, Streamlit, Plotly, Scikit-Learn, Scipy, Pandas.
