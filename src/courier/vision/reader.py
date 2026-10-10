"""Đọc một ảnh bản đồ -> thông tin bản đồ (giống scenes.json) -> Scene cho khối 3.

Các bước:
  1. Mạng dò điểm (KeypointNet) trên ảnh thu nhỏ 512: tâm giao lộ, 3 ô mẫu chú giải (thường/đông/mái che),
     ô thời tiết.
  2. Dựng lưới: ước lượng góc xoay, gom tâm giao lộ thành hàng / cột -> (hàng, cột) cho từng giao lộ.
  3. Mỗi cặp giao lộ kề nhau: cắt dải ảnh dọc đoạn nối -> EdgeNet: không có đường / kiểu nhìn / đóng,
     bậc thang, một chiều + chiều.
  4. Mỗi giao lộ: cắt ô vuông -> NodeNet: trống / loại địa điểm / robot (+ hướng mũi).
  5. Ô mẫu chú giải -> MiscNet: mỗi ô vẽ theo kiểu nhìn nào -> cách hoán đổi (road_look) của CHÍNH ảnh này.
     Ô thời tiết -> MiscNet: mưa / khô.
  6. Đổi "kiểu nhìn" của từng đoạn sang trạng thái thật bằng road_look.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import common as C
from .models import KeypointNet, edge_net, node_net, misc_net, n_params
from ..graph import Scene, Edge, Mission, HEADING

MODELS = Path(__file__).resolve().parents[3] / "models"
NODE_THRESHOLD = 0.05      # ngưỡng bản đồ nhiệt cho tâm giao lộ (chọn trên val-A)
# bước 5C: đỉnh yếu (LOW..NODE_THRESHOLD) được hậu kiểm bằng cấu trúc lưới; None = tắt. Đặt bằng COURIER_LOW_THR.
LOW_THRESHOLD = (lambda v: None if v in ("", "none") else float(v))(__import__("os").environ.get("COURIER_LOW_THR", "0.02"))


class VisionReader:
    def __init__(self, models_dir=MODELS, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg_path = Path(models_dir) / "config.json"                  # bản 2: nền + độ phân giải đầu vào
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.kp_input = cfg.get("kp_input", C.KP_INPUT)
        self.node_threshold = cfg.get("node_threshold", NODE_THRESHOLD)
        self.low_threshold = cfg.get("low_threshold", LOW_THRESHOLD)
        self.tta_shift = cfg.get("tta_shift", float(__import__("os").environ.get("COURIER_TTA", "0")))
        self.kp = KeypointNet(pretrained=False, backbone=cfg.get("backbone", "resnet18"))
        self.edge, self.node, self.misc = edge_net(), node_net(), misc_net()
        for name, m in [("kp", self.kp), ("edge", self.edge), ("node", self.node), ("misc", self.misc)]:
            m.load_state_dict(torch.load(Path(models_dir) / f"{name}.pt", map_location="cpu", weights_only=True))
            m.to(self.dev).eval()

    def n_params(self):
        return {k: n_params(m) for k, m in [("kp", self.kp), ("edge", self.edge), ("node", self.node),
                                            ("misc", self.misc)]}

    # ------------------------------------------------------------- 1. dò điểm
    @torch.no_grad()
    def keypoints(self, im):
        _, H, W = im.shape
        S = self.kp_input
        sc = S / max(H, W)
        x = torch.zeros(1, 3, S, S, device=self.dev)
        r = F.interpolate(im[None], size=(round(H * sc), round(W * sc)), mode="bilinear", align_corners=False)
        x[0, :, :r.shape[2], :r.shape[3]] = r[0]
        hm = torch.sigmoid(self.kp(x)[0].float())                      # (5,R,R)

        def to_img(cy, cx):
            return ((cx + 0.5) * C.KP_STRIDE / sc, (cy + 0.5) * C.KP_STRIDE / sc)

        def refine(ch, y, x_):
            y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, hm.shape[1]), max(x_ - 1, 0), min(x_ + 2, hm.shape[2])
            w = hm[ch, y0:y1, x0:x1]
            yy, xx = torch.meshgrid(torch.arange(y0, y1, device=self.dev), torch.arange(x0, x1, device=self.dev),
                                    indexing="ij")
            s = w.sum()
            return float((w * yy).sum() / s), float((w * xx).sum() / s)

        peak = (hm == F.max_pool2d(hm[None], 3, 1, 1)[0])
        lo = self.node_threshold if self.low_threshold is None else min(self.low_threshold, self.node_threshold)
        ys, xs = torch.nonzero(peak[0] & (hm[0] > lo), as_tuple=True)
        nodes = []
        for y, x_ in zip(ys.tolist(), xs.tolist()):
            cy, cx = refine(0, y, x_)
            nodes.append((*to_img(cy, cx), float(hm[0, y, x_])))
        singles = []
        for ch in range(1, 5):
            k = int(torch.argmax(hm[ch]))
            y, x_ = divmod(k, hm.shape[2])
            cy, cx = refine(ch, y, x_)
            singles.append((*to_img(cy, cx), float(hm[ch, y, x_])))
        return nodes, singles

    # ------------------------------------------------------------- 2. dựng lưới
    @staticmethod
    def build_grid(nodes):
        P = np.array([(x, y) for x, y, _ in nodes], dtype=np.float64)
        score = np.array([s for _, _, s in nodes])
        if len(P) < 4:
            return None
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        np.fill_diagonal(D, np.inf)
        nn_d = D.min(1)
        spacing = float(np.median(nn_d))

        def cluster(v):
            order = np.argsort(v)
            lab = np.zeros(len(v), dtype=int)
            k = 0
            for a, b in zip(order[:-1], order[1:]):
                if v[b] - v[a] > 0.45 * spacing:
                    k += 1
                lab[b] = k
            return lab, k + 1

        def spread(v):                      # tổng độ trải trong từng nhóm + phạt mỗi nhóm thêm (tránh tách đôi hàng)
            lab, n = cluster(v)
            return sum(np.ptp(v[lab == k]) for k in range(n)) + 0.5 * spacing * n

        def rotate(deg):
            t = math.radians(deg)
            c, s = math.cos(t), math.sin(t)
            Z = P - P.mean(0)
            return np.stack([c * Z[:, 0] + s * Z[:, 1], -s * Z[:, 0] + c * Z[:, 1]], 1)

        # góc xoay: thử -4..4 độ, chọn góc làm các hàng + cột "gọn" nhất (tổng độ trải trong mỗi hàng/cột nhỏ nhất)
        cands = np.arange(-4.0, 4.0001, 0.1)
        costs = [spread(rotate(d)[:, 1]) + spread(rotate(d)[:, 0]) for d in cands]
        deg = float(cands[int(np.argmin(costs))])
        Q = rotate(deg)
        th = math.radians(deg)

        rlab, nr = cluster(Q[:, 1])
        clab, nc = cluster(Q[:, 0])
        best = {}
        for i in range(len(P)):
            rc = (int(rlab[i]), int(clab[i]))
            if rc not in best or score[i] > score[best[rc]]:
                best[rc] = i
        row_c = [float(np.median(Q[rlab == k, 1])) for k in range(nr)]
        col_c = [float(np.median(Q[clab == k, 0])) for k in range(nc)]
        return {"rows": nr, "cols": nc, "xy": {rc: tuple(P[i]) for rc, i in best.items()}, "angle": math.degrees(th),
                "deg": deg, "mean": P.mean(0), "row_c": row_c, "col_c": col_c, "spacing": spacing}

    @staticmethod
    def recover_weak(g, weak, tol=0.3):
        """Hậu kiểm bằng cấu trúc lưới (bước 5C): điểm YẾU (đỉnh nhiệt dưới ngưỡng chính) chỉ được nhận khi nằm đúng
        giao của một hàng và một cột đã dựng từ các điểm chắc chắn (sai lệch < tol x khoảng cách giao lộ) và ô đó
        đang trống. Điểm thừa (góc ảnh, ký hiệu trên đường, chữ) thường không thỏa cả hai điều kiện."""
        t = math.radians(g["deg"])
        c, s_ = math.cos(t), math.sin(t)
        occupied = set(g["xy"])
        out = []
        for x, y, sc in sorted(weak, key=lambda n: -n[2]):
            zx, zy = x - g["mean"][0], y - g["mean"][1]
            qx, qy = c * zx + s_ * zy, -s_ * zx + c * zy
            r = int(np.argmin([abs(qy - v) for v in g["row_c"]]))
            k = int(np.argmin([abs(qx - v) for v in g["col_c"]]))
            if (abs(qy - g["row_c"][r]) < tol * g["spacing"] and abs(qx - g["col_c"][k]) < tol * g["spacing"]
                    and (r, k) not in occupied):
                occupied.add((r, k))
                out.append((x, y, sc))
        return out

    def _tta(self, net, crop, n):
        """Suy luận nhiều lần rồi gộp (bước 5C): cắt mảnh tại chỗ + lệch nhẹ TTA_SHIFT pixel theo 4 hướng chéo,
        lấy TRUNG BÌNH xác suất. TTA_SHIFT = 0 -> chỉ cắt một lần như cũ."""
        offs = [(0.0, 0.0)] + ([(d, d), (-d, -d), (d, -d), (-d, d)] if (d := self.tta_shift) else [])
        acc = None
        for ox, oy in offs:
            o = torch.tensor([[ox, oy]], dtype=torch.float32, device=self.dev)
            out = {k: F.softmax(v.float(), 1) for k, v in net(crop(o)).items()}
            acc = out if acc is None else {k: acc[k] + out[k] for k in acc}
        return {k: v / len(offs) for k, v in acc.items()}

    # ------------------------------------------------------------- toàn bộ
    @torch.no_grad()
    def read(self, path):
        img = path if hasattr(path, "convert") else C.load_rgb(path)     # nhận đường dẫn hoặc ảnh PIL
        im = C.to_tensor(img).to(self.dev)
        nodes, singles = self.keypoints(im)
        strong = [n for n in nodes if n[2] > self.node_threshold]
        g = self.build_grid(strong)
        if g is None:
            return None
        if self.low_threshold is not None:            # hậu kiểm bằng cấu trúc lưới: cứu giao lộ có đỉnh nhiệt yếu
            extra = self.recover_weak(g, [n for n in nodes if n[2] <= self.node_threshold])
            if extra:
                g = self.build_grid(strong + extra)
                nodes = strong + extra
            else:
                nodes = strong
        else:
            nodes = strong
        xy = g["xy"]
        rcs = sorted(xy)
        T = torch.tensor([xy[rc] for rc in rcs], dtype=torch.float32, device=self.dev)
        idx = {rc: k for k, rc in enumerate(rcs)}

        # 3. đoạn đường
        pairs = [(rc, nb) for rc in rcs for nb in ((rc[0], rc[1] + 1), (rc[0] + 1, rc[1])) if nb in idx]
        edges = []
        if pairs:
            a = T[[idx[p[0]] for p in pairs]]
            b = T[[idx[p[1]] for p in pairs]]
            oe = self._tta(self.edge, lambda o: C.edge_patches(im, a + o, b + o), len(pairs))
            look = oe["look"].cpu().numpy()
            stairs = oe["stairs"].argmax(1).cpu().numpy()
            ow = oe["oneway"].argmax(1).cpu().numpy()
            for k, (p, q) in enumerate(pairs):
                edges.append({"a": p, "b": q, "look": int(look[k].argmax()), "p": look[k],
                              "stairs": bool(stairs[k]), "oneway": int(ow[k])})

        # 4. giao lộ
        on = self._tta(self.node, lambda o: C.square_patches(im, T + o, C.NODE_REGION, C.NODE_SIZE), len(rcs))
        pc = on["content"].cpu().numpy()
        ph = on["heading"].cpu().numpy()
        ROBOT = len(C.NODE_CLASSES) - 1
        r_idx = int(pc[:, ROBOT].argmax())
        landmarks = {}
        for k, rc in enumerate(rcs):
            if k == r_idx:
                continue
            cls = int(pc[k, :ROBOT].argmax())
            if cls > 0:
                landmarks.setdefault(C.PLACE_TYPES[cls - 1], []).append(rc)

        # 5. chú giải + thời tiết
        leg = torch.tensor([s_[:2] for s_ in singles[:3]], dtype=torch.float32, device=self.dev)
        pl = F.softmax(self.misc(C.square_patches(im, leg, C.LEG_REGION, C.MISC_SIZE))["legend"].float(), 1)
        pl = pl.cpu().numpy()                                          # pl[kind, look]
        best, road_look = -1, None
        for perm in itertools.permutations(range(3)):
            v = sum(math.log(pl[k, perm[k]] + 1e-9) for k in range(3))
            if v > best or road_look is None:
                best, road_look = v, {C.LOOKS[k]: C.LOOKS[perm[k]] for k in range(3)}
        wc = torch.tensor([singles[3][:2]], dtype=torch.float32, device=self.dev)
        pw = F.softmax(self.misc(C.square_patches(im, wc, C.WEATHER_REGION, C.MISC_SIZE))["weather"].float(), 1)
        rain = bool(pw[0, 1] > 0.5)

        # 6. kiểu nhìn -> trạng thái thật
        look2status = {look: st for st, look in road_look.items()}
        out_edges = []
        for e in edges:
            if e["look"] == 0:
                continue
            status = "closed" if e["look"] == 4 else look2status[C.LOOKS[e["look"] - 1]]
            oneway_to = None if e["oneway"] == 0 else (e["b"] if e["oneway"] == 1 else e["a"])
            out_edges.append({"a": e["a"], "b": e["b"], "status": status, "stairs": e["stairs"],
                              "oneway_to": oneway_to})
        return {
            "grid": (g["rows"], g["cols"]), "nodes": rcs, "edges": out_edges, "landmarks": landmarks,
            "robot": rcs[r_idx], "heading": C.HEADINGS[int(ph[r_idx].argmax())], "rain": rain,
            "road_look": road_look, "angle": g["angle"],
            "xy": {rc: xy[rc] for rc in rcs}, "raw_nodes": nodes,     # để phân tích lỗi (không dùng khi dự đoán)
        }


def scene_from_vision(v, mission=None):
    """Kết quả read() -> Scene (đúng loại khối 3 dùng). mission: Mission hoặc None (gán sau)."""
    edges = {frozenset((e["a"], e["b"])): Edge(e["status"], e["stairs"], e["oneway_to"]) for e in v["edges"]}
    return Scene(rows=v["grid"][0], cols=v["grid"][1], nodes=set(v["nodes"]), edges=edges,
                 landmarks={k: list(x) for k, x in v["landmarks"].items()}, robot=tuple(v["robot"]),
                 heading=HEADING[v["heading"]], rain=v["rain"], mission=mission or Mission(goal=None))
