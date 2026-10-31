# Bài tập lớn 2 — Thị giác máy tính

Repository gồm hai notebook thực nghiệm về **ghép đối tượng vào ảnh nền** và **ghép ảnh toàn cảnh (panorama)**. Mỗi notebook trình bày thuật toán, mã nguồn, kết quả trung gian và nhận xét để theo dõi toàn bộ quá trình xử lý.

| Nội dung | Notebook | Phương pháp so sánh |
|---|---|---|
| Gradient Domain Editing | [BTL2CV_GradientDomainEditing.ipynb](BTL2CV_GradientDomainEditing.ipynb) | Direct paste, Feather, Laplacian pyramid, Poisson blending |
| Ghép ảnh toàn cảnh | [panorama_sift_orb.ipynb](panorama_sift_orb.ipynb) | SIFT, ORB, SURF |

## 1. Cấu trúc thư mục

```text
BTL2CV/
├── README.md
├── requirements.txt
├── BTL2CV_GradientDomainEditing.ipynb
├── panorama_sift_orb.ipynb
├── anhnhanvat.jpg                  # Ảnh nguồn cho bài ghép đối tượng
├── background.jpg                  # Ảnh nền cho bài ghép đối tượng
├── BK-H6-imgs/                     # 4 ảnh đầu vào cho panorama
└── company/                        # 5 ảnh; notebook sử dụng 4 ảnh đầu theo tên
```

## 2. Cài đặt và chạy

### Môi trường

Hai notebook đã được chạy kiểm tra với **Python 3.11** và các phiên bản trong [requirements.txt](requirements.txt). Các phiên bản NumPy và SciPy đang được cố định trong file này yêu cầu Python từ 3.11.

| Thư viện | Vai trò |
|---|---|
| NumPy | Tính toán trên mảng và ảnh |
| OpenCV (`opencv-python`) | Đọc ảnh, biến đổi hình học, SIFT/ORB, matching và blending |
| SciPy | Xây dựng và giải hệ phương trình thưa cho Poisson blending |
| Mahotas | Trích xuất đặc trưng SURF |
| IPython, ipykernel | Hiển thị kết quả và chạy kernel notebook |

### Tải mã nguồn và cài thư viện

```bash
git clone https://github.com/hoangdelima09/BTL2CV.git
cd BTL2CV
python3.11 -m venv .venv
```

Kích hoạt môi trường trên Linux/macOS:

```bash
source .venv/bin/activate
```

Trên Windows PowerShell, dùng `py -3.11 -m venv .venv` để tạo môi trường, sau đó kích hoạt bằng:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài thư viện và đăng ký kernel:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name btl2cv --display-name "Python (BTL2CV)"
```

### Mở notebook

**Với VS Code:** mở thư mục `BTL2CV`, cài extension Python và Jupyter, sau đó mở notebook và chọn kernel **Python (BTL2CV)** hoặc interpreter trong `.venv`.

**Với JupyterLab:** cài thêm giao diện JupyterLab trong môi trường đã kích hoạt:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

Chọn kernel đã đăng ký và chạy các cell **theo thứ tự từ trên xuống**, hoặc dùng **Restart Kernel and Run All**. Hai notebook chạy độc lập, có thể mở notebook nào trước cũng được.

> Các notebook tìm ảnh từ `Path.cwd()`. Hãy chạy với thư mục làm việc là thư mục gốc `BTL2CV`, nơi chứa hai notebook và các ảnh đầu vào. Nếu không tìm thấy ảnh, kiểm tra `Path.cwd()` trong một cell.

## 3. Gradient Domain Editing

Notebook [BTL2CV_GradientDomainEditing.ipynb](BTL2CV_GradientDomainEditing.ipynb) đặt đối tượng từ `anhnhanvat.jpg` lên `background.jpg`, sau đó so sánh bốn cách ghép với cùng mask và vị trí:

| Phương pháp | Cách xử lý |
|---|---|
| Direct paste | Thay trực tiếp pixel trong mask bằng pixel nguồn |
| Feather (alpha) | Làm mềm vùng biên và trộn ảnh bằng trọng số alpha |
| Laplacian pyramid | Trộn ảnh theo các dải tần số của tháp Laplacian |
| Poisson blending | Giải hệ phương trình thưa để giữ gradient nguồn, với giá trị nền làm điều kiện biên |

**Các bước thực hiện:** đọc ảnh → tạo mask → thu phóng và đặt đối tượng → chạy bốn phương pháp → so sánh ảnh toàn cảnh và vùng biên phóng to.

Với ảnh mẫu, notebook dùng đa giác có sẵn để tạo mask. Khi thay ảnh nguồn, có thể cung cấp file mask hoặc dùng GrabCut với khung chữ nhật bao quanh đối tượng.

Các tham số chính ở đầu notebook:

| Biến / tham số | Ý nghĩa |
|---|---|
| `BACKGROUND`, `SOURCE` | Đường dẫn ảnh nền và ảnh nguồn |
| `CONFIG["mask"]` | Đường dẫn mask có cùng kích thước ảnh nguồn; `None` nếu không dùng |
| `CONFIG["rect"]` | Khung `(x, y, w, h)` cho GrabCut; cần chừa một phần nền ngoài khung |
| `CONFIG["scale"]` | Hệ số thu phóng ảnh nguồn; mặc định `1.0` với ảnh mẫu |
| `CONFIG["pos"]` | Vị trí góc trên trái `(x, y)` của ảnh nguồn trên nền; `None` để tự chọn |

**Kết quả trong notebook:** ảnh nguồn và nền, mask, đối tượng sau khi đặt lên canvas, bốn ảnh ghép, lưới so sánh và ảnh phóng to biên.

Với cặp ảnh mẫu, Poisson blending làm liền biên nhưng cũng làm thay đổi mạnh màu và độ sáng của nhân vật. Chất lượng cần được đánh giá đồng thời ở đường nối, chi tiết và màu sắc.

## 4. Ghép ảnh toàn cảnh

Notebook [panorama_sift_orb.ipynb](panorama_sift_orb.ipynb) so sánh ba phương pháp trích xuất đặc trưng **SIFT, ORB và SURF** trên cùng quy trình:

```text
Đọc ảnh và giảm kích thước
    → Chiếu ảnh lên mặt trụ
    → Trích xuất keypoint và descriptor
    → Matching các cặp ảnh kề nhau + ratio test
    → Ước lượng homography bằng RANSAC
    → Đưa ảnh về hệ tọa độ chung
    → Feather blending và đánh giá
```

SIFT và ORB được tính bằng OpenCV; SURF được tính bằng `mahotas.features.surf`. Matching dùng khoảng cách **L2** cho SIFT/SURF và **Hamming** cho ORB. Notebook không cần `cv.xfeatures2d.SURF_create`.

### Dữ liệu và cấu hình mặc định

- `BK-H6-imgs/`: sử dụng cả 4 ảnh `.jpg`, sắp theo tên file.
- `company/`: có 5 ảnh `.jfif`, sử dụng 4 ảnh đầu theo tên file.

| Tham số trong `CONFIG` | Mặc định | Ý nghĩa |
|---|---:|---|
| `max_side` | `900` | Cạnh dài tối đa của ảnh sau thu nhỏ |
| `nfeatures` | `2500` | Tham số số lượng đặc trưng truyền cho bộ trích xuất |
| `ratio` | `0.78` | Ngưỡng ratio test khi lọc matching |
| `ransac_px` | `4.0` | Ngưỡng sai số RANSAC, tính bằng pixel trên ảnh đã xử lý |
| `focal_ratio` | `1.2` | Tiêu cự ước lượng bằng hệ số này nhân chiều rộng ảnh |

Khi thử bộ ảnh khác, sửa `DATASETS` ở đầu notebook. Các ảnh cần có vùng chồng lấn và được xếp theo thứ tự góc nhìn để ghép các cặp kề nhau.

### Kết quả và đánh giá

Notebook hiển thị ảnh đầu vào, ảnh sau phép chiếu, keypoint, match và inlier RANSAC, overlay kiểm tra căn chỉnh, coverage map, ảnh ghép trung bình đều và panorama sau feather blending.

Bảng số liệu gồm số keypoint, match, inlier, tỉ lệ inlier, sai số chiếu lại trung vị và thời gian xử lý. Khi so sánh, cần xem cả số liệu lẫn độ liên tục của kiến trúc, bóng đôi, đường nối và màu sắc trên panorama.

## 5. Giới hạn thực nghiệm

- **Ghép đối tượng:** phụ thuộc vào chất lượng mask và độ chênh màu giữa ảnh nguồn với nền. Poisson blending không ràng buộc giữ nguyên màu tuyệt đối; mask lớn cũng làm tăng chi phí giải hệ phương trình.
- **Panorama:** vật thể chuyển động và parallax có thể tạo bóng đôi hoặc méo hình. Feather blending làm mềm đường nối nhưng không sửa được sai lệch hình học.
- **Phép chiếu mặt trụ:** tiêu cự được ước lượng, chưa sử dụng hiệu chuẩn máy ảnh.
- **Vùng trống:** panorama có thể còn góc đen vì phép cắt hiện tại giữ toàn bộ vùng có dữ liệu.
- **Đánh giá:** kết quả áp dụng cho các bộ ảnh mẫu; thời gian chạy thay đổi theo máy. So sánh tốc độ SIFT/ORB/SURF còn chịu ảnh hưởng của khác biệt triển khai giữa OpenCV và Mahotas.

Ảnh kết quả và bảng đánh giá được hiển thị trực tiếp trong các notebook. Hãy lưu notebook sau khi chạy nếu muốn giữ lại output của lần thực nghiệm mới.
