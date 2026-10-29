import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized

BACKGROUND = "background.jpg"
SOURCE = "anhnhanvat.jpg"
SCALE = 1.0 if Path(SOURCE).name == "anhnhanvat.jpg" else 0.35

MASK_SIZE = (1200, 675)
MASK_POINTS = [
    (515, 0), (750, 0), (747, 9), (738, 20), (781, 25), (769, 42), (795, 51), (782, 67),
    (805, 82), (792, 93), (814, 106), (793, 115), (808, 126), (786, 131), (791, 144), (774, 146),
    (777, 164), (764, 160), (763, 191), (771, 194), (771, 218), (764, 246), (749, 276), (725, 296),
    (724, 305), (754, 309), (791, 322), (820, 326), (835, 339), (841, 364), (854, 377), (863, 403),
    (887, 416), (896, 440), (927, 451), (952, 468), (973, 493), (993, 528), (1008, 562), (1026, 611),
    (1041, 650), (1045, 674), (118, 674), (116, 643), (113, 616), (122, 583), (141, 540), (160, 500),
    (183, 457), (204, 415), (225, 376), (239, 357), (256, 351), (271, 346), (292, 315), (318, 281),
    (344, 253), (365, 235), (393, 214), (423, 193), (452, 169), (480, 150), (463, 140), (451, 127),
    (469, 119), (448, 98), (478, 91), (457, 80), (480, 67), (465, 54), (490, 46), (474, 32),
    (500, 29), (487, 19), (525, 20),
]
MASK_HOLE = [
    (493, 204), (479, 208), (471, 218), (462, 229), (449, 238), (441, 243), (439, 257), (437, 270),
    (437, 283), (453, 276), (470, 266), (483, 258), (495, 257), (503, 261), (505, 268), (502, 277),
    (488, 287), (470, 301), (449, 317), (431, 334), (412, 353), (393, 368), (377, 374), (356, 377),
    (339, 374), (343, 393), (348, 414), (344, 439), (365, 439), (395, 442), (422, 447), (420, 440),
    (421, 429), (439, 419), (458, 410), (466, 396), (469, 379), (470, 361), (477, 350), (496, 344),
    (521, 337), (542, 326), (534, 314), (528, 299), (524, 287), (514, 275), (505, 257), (499, 244),
    (495, 229), (493, 213),
]


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Đọc ảnh từ file."""
    image = cv2.imdecode(np.fromfile(path, np.uint8), flags) if path.is_file() else None
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def write_image(path, image):
    """Ghi ảnh ra file."""
    cv2.imencode(path.suffix or ".png", image)[1].tofile(path)


def binarize(mask):
    """Nhị phân hóa mask về 0/255."""
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def parse_ints(value, count):
    """Đọc chuỗi "a,b,..." thành bộ số nguyên."""
    parts = tuple(map(int, value.split(",")))
    if len(parts) != count:
        raise ValueError(f"Cần {count} số nguyên, phân cách bằng dấu phẩy.")
    return parts


def click_polygon_mask(source):
    """Tạo mask thủ công bằng cách click các đỉnh đa giác quanh đối tượng."""
    view = min(1.0, 900 / source.shape[0], 1400 / source.shape[1])
    shown = cv2.resize(source, None, fx=view, fy=view) if view < 1 else source
    title = "Click quanh doi tuong | Chuot phai/Backspace: xoa diem | Enter: xong | Esc: huy"
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        canvas = shown.copy()
        if points:
            cv2.polylines(canvas, [np.int32(points)], len(points) >= 3, (0, 255, 0), 1)
            for p in points:
                cv2.circle(canvas, p, 3, (0, 0, 255), -1)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):
            break
        if key == 27:
            points.clear()
            break
        if key == 8 and points:
            points.pop()
    cv2.destroyAllWindows()
    if len(points) < 3:
        raise ValueError("Cần click ít nhất 3 điểm để tạo mask.")
    mask = np.zeros(source.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(np.array(points) / view)], 255)
    return mask


def grabcut_mask(source, rect):
    """Tạo mask bằng GrabCut từ khung chữ nhật quanh đối tượng."""
    x, y, w, h = rect
    sh, sw = source.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > sw or y + h > sh or w * h == sw * sh:
        raise ValueError("Khung GrabCut phải nằm trong ảnh và chừa lại một phần nền bên ngoài.")
    gc = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(source, gc, tuple(rect), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def polygon_mask(source):
    """Tạo mask nhân vật từ đa giác vẽ tay có sẵn."""
    if source.shape[1::-1] != MASK_SIZE:
        raise ValueError("Kích thước ảnh nguồn không khớp mask có sẵn.")
    mask = np.zeros(source.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(MASK_POINTS)], 255)
    cv2.fillPoly(mask, [np.int32(MASK_HOLE)], 0)
    return mask


def place_on_target(source, mask, target, top_left):
    """Đặt ảnh nguồn và mask lên khung ảnh nền."""
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    x0, y0 = top_left
    tx0, ty0 = max(x0, 0), max(y0, 0)
    tx1, ty1 = min(x0 + sw, tw), min(y0 + sh, th)
    if tx0 >= tx1 or ty0 >= ty1:
        raise ValueError("Ảnh nguồn nằm hoàn toàn ngoài ảnh nền.")

    src = target.copy()
    msk = np.zeros((th, tw), np.uint8)
    src[ty0:ty1, tx0:tx1] = source[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    msk[ty0:ty1, tx0:tx1] = mask[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    msk = binarize(msk)
    msk[[0, -1], :] = msk[:, [0, -1]] = 0
    if not msk.any():
        raise ValueError("Mask rỗng sau khi đặt lên nền; kiểm tra mask, --pos và --scale.")
    return src, msk


def poisson_blend(target, source, mask, mixed=False, screen_weight=0.0):
    """Ghép ảnh bằng Poisson (import, mixed hoặc screened)."""
    if mask[[0, -1], :].any() or mask[:, [0, -1]].any():
        raise ValueError("Mask Poisson phải chừa biên ngoài canvas một pixel.")
    f_star = target.astype(np.float64)
    g = source.astype(np.float64)

    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return target.copy()
    index = np.full(mask.shape, -1)
    index[ys, xs] = np.arange(n)

    rows, cols = [], []
    b = screen_weight * g[ys, xs]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        qy, qx = ys + dy, xs + dx

        v = g[ys, xs] - g[qy, qx]
        if mixed:
            v_dst = f_star[ys, xs] - f_star[qy, qx]
            use_dst = (v_dst ** 2).sum(axis=1) > (v ** 2).sum(axis=1)
            v[use_dst] = v_dst[use_dst]
        b += v

        q = index[qy, qx]
        inside = q >= 0
        rows.append(np.nonzero(inside)[0])
        cols.append(q[inside])
        b[~inside] += f_star[qy[~inside], qx[~inside]]

    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adjacency = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (4.0 + screen_weight) * sparse.identity(n, format="csc") - adjacency
    solve = factorized(A)
    result = f_star.copy()
    result[ys, xs] = np.column_stack([solve(b[:, c]) for c in range(3)])
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def direct_paste(target, source, mask):
    """Dán trực tiếp ảnh nguồn lên ảnh nền."""
    return np.where(mask[..., None] > 0, source, target)


def feather_blend(target, source, mask, sigma=5.0):
    """Ghép bằng alpha blending với biên mask làm mờ."""
    kernel = np.ones((int(4 * sigma) + 1,) * 2, np.uint8)
    alpha = cv2.erode(mask, kernel).astype(np.float64) / 255
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma) * (mask > 0)
    alpha = alpha[..., None]
    return (alpha * source + (1 - alpha) * target).astype(np.uint8)


def pyramid_blend(target, source, mask, levels=5):
    """Ghép bằng Laplacian pyramid."""
    ga = [source.astype(np.float32)]
    gb = [target.astype(np.float32)]
    gm = [cv2.merge([mask.astype(np.float32) / 255] * 3)]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        gm.append(cv2.pyrDown(gm[-1]))

    blended = gm[-1] * ga[-1] + (1 - gm[-1]) * gb[-1]
    for i in reversed(range(levels)):
        size = ga[i].shape[1::-1]
        la = ga[i] - cv2.pyrUp(ga[i + 1], dstsize=size)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        blended = cv2.pyrUp(blended, dstsize=size) + gm[i] * la + (1 - gm[i]) * lb
    return np.clip(blended, 0, 255).astype(np.uint8)


def labeled(image, text):
    """Gắn nhãn lên góc trên ảnh."""
    tile = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, tile.shape[1] / 900)
    thick = max(1, round(2 * scale))
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(tile, (0, 0), (w + 12, h + base + 12), (0, 0, 0), -1)
    cv2.putText(tile, text, (6, h + 6), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return tile


def make_grid(tiles, cols, gap=6):
    """Xếp các ảnh thành lưới."""
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * (h + gap) - gap, cols * (w + gap) - gap, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (h + gap), c * (w + gap)
        grid[y:y + h, x:x + w] = tile
    return grid


def zoom_box(msk):
    """Chọn khung phóng to quanh biên vùng ghép."""
    x, y, w, h = cv2.boundingRect(msk)
    cy = y + round(0.68 * (h - 1))
    row = np.flatnonzero(msk[cy])
    cx = int(row[-1]) if row.size else x + w // 2
    pad = max(16, min(w, h) // 8)
    x0, y0 = max(cx - pad, 0), max(cy - pad, 0)
    x1, y1 = min(cx + pad + 1, msk.shape[1]), min(cy + pad + 1, msk.shape[0])
    return x0, y0, x1, y1


def parse_args():
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Gradient Domain Editing")
    parser.add_argument("--bg", type=Path, default=Path(BACKGROUND), help="ảnh nền")
    parser.add_argument("--src", type=Path, default=Path(SOURCE), help="ảnh nguồn chứa đối tượng")
    parser.add_argument("--mask", type=Path, help="mask cùng kích thước nguồn; ảnh mẫu dùng polygon có sẵn")
    parser.add_argument("--rect", help="khung x,y,w,h cho GrabCut ở tọa độ nguồn gốc")
    parser.add_argument("--pos", help="góc trên trái nguồn sau scale; ảnh mẫu lệch trái, ảnh khác ở giữa")
    parser.add_argument("--scale", type=float, default=SCALE, help="hệ số thu phóng; bộ ảnh mẫu mặc định vừa chiều cao nền")
    parser.add_argument("--screen-weight", type=float, default=0.05, help="lambda của screened Poisson (mặc định 0.05)")
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args()
    if not np.isfinite(args.screen_weight) or args.screen_weight < 0:
        raise ValueError("--screen-weight phải hữu hạn và >= 0.")
    if args.mask and args.rect:
        raise ValueError("Chỉ chọn một trong --mask và --rect.")
    return args


def load_mask(args, source, use_preset):
    """Tạo mask cho ảnh nguồn."""
    if args.mask:
        mask = read_image(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask.shape != source.shape[:2]:
            raise ValueError("Mask phải cùng kích thước với ảnh nguồn gốc.")
    elif use_preset:
        mask = polygon_mask(source)
    elif args.rect:
        mask = grabcut_mask(source, parse_ints(args.rect, 4))
    else:
        mask = click_polygon_mask(source)
    mask = binarize(mask)
    if not mask.any():
        raise ValueError("Mask không chứa đối tượng.")
    return mask


def resize_source(source, mask, scale):
    """Thu phóng ảnh nguồn và mask."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("--scale phải là số hữu hạn > 0.")
    size = (round(source.shape[1] * scale), round(source.shape[0] * scale))
    if min(size) < 1:
        raise ValueError("--scale quá nhỏ.")
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(source, size, interpolation=interp), cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def choose_position(pos, source, mask, target, use_preset):
    """Chọn vị trí đặt ảnh nguồn lên ảnh nền."""
    if pos:
        return parse_ints(pos, 2)
    if use_preset:
        return (-round(0.275 * source.shape[1]), target.shape[0] - source.shape[0])
    x, y, w, h = cv2.boundingRect(mask)
    return (target.shape[1] // 2 - x - w // 2, target.shape[0] // 2 - y - h // 2)


def blend_all(target, src, msk, screen_weight):
    """Chạy tất cả phương pháp ghép."""
    return {
        "1_direct": ("Direct paste", direct_paste(target, src, msk)),
        "2_feather": ("Feather (alpha)", feather_blend(target, src, msk)),
        "3_pyramid": ("Laplacian pyramid", pyramid_blend(target, src, msk)),
        "4_poisson_import": ("Poisson import", poisson_blend(target, src, msk)),
        "5_poisson_mixed": ("Poisson mixed", poisson_blend(target, src, msk, mixed=True)),
        "6_poisson_screened": ("Screened Poisson", poisson_blend(target, src, msk, screen_weight=screen_weight)),
    }


def save_comparison(out, target, src, msk, results):
    """Lưu ảnh lưới so sánh các phương pháp."""
    source_view = src.copy()
    contours, _ = cv2.findContours(msk, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(source_view, contours, -1, (0, 255, 0), 2)
    tiles = [labeled(target, "Background"), labeled(source_view, "Source + mask")]
    tiles += [labeled(image, label) for label, image in results.values()]
    write_image(out / "comparison.png", make_grid(tiles, cols=4))


def save_zoom(out, msk, results):
    """Lưu ảnh phóng to vùng biên của các phương pháp."""
    x0, y0, x1, y1 = zoom_box(msk)
    zoom_size = (round(320 * (x1 - x0) / (y1 - y0)), 320)
    zooms = [labeled(cv2.resize(image[y0:y1, x0:x1], zoom_size, interpolation=cv2.INTER_NEAREST), label)
             for label, image in results.values()]
    write_image(out / "zoom.png", make_grid(zooms, cols=len(zooms)))


def run_all():
    """Chạy toàn bộ quy trình và lưu kết quả so sánh."""
    args = parse_args()
    target = read_image(args.bg)
    original_source = read_image(args.src)
    use_preset = args.src.name == "anhnhanvat.jpg" and not (args.mask or args.rect)

    source_mask = load_mask(args, original_source, use_preset)
    if args.scale is not None:
        scale = args.scale
    else:
        scale = target.shape[0] / original_source.shape[0] if use_preset else 1.0
    source, mask = resize_source(original_source, source_mask, scale)
    top_left = choose_position(args.pos, source, mask, target, use_preset)
    src, msk = place_on_target(source, mask, target, top_left)
    results = blend_all(target, src, msk, args.screen_weight)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_image(out / "source_mask.png", source_mask)
    write_image(out / "mask.png", msk)
    for name, (_, image) in results.items():
        write_image(out / f"{name}.png", image)
    save_comparison(out, target, src, msk, results)
    save_zoom(out, msk, results)
    print(f"Đã lưu kết quả vào {out}")


run_all()
