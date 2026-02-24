import os
import csv
import json
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageFont
import multiprocessing
import numpy as np
import warnings
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import copy
import random
import threading
import queue as _queue

# 警告抑制
warnings.filterwarnings("ignore")

# --------- 設定 ---------
# 検出用モデルはオリジナルの3クラス (bg + void + crack)
LABEL_NAMES_DETECT = ['void', 'crack']
NUM_CLASSES_DETECT = len(LABEL_NAMES_DETECT) + 1
ID2LABEL_DETECT = {idx + 1: label for idx, label in enumerate(LABEL_NAMES_DETECT)}
LABEL2ID_DETECT = {label: idx + 1 for idx, label in enumerate(LABEL_NAMES_DETECT)}

SCORE_THRESH = 0.85
RAW_SCORE_THRESH = 0.30  # raw保持用の低閾値 (スライダーで表示切替)
INPUT_SIZE = (640, 640)

# レビュー用ラベル (GUIで選択可能)
REVIEW_LABELS = ['void', 'crack', 'other']

# レビューGUI用の色定義 (目に優しい低彩度)
LABEL_COLORS = {
    'void': '#C86464',       # くすんだ赤
    'crack': '#6496C8',      # くすんだ青
    'other': '#808080',      # グレー
    'unlabeled': '#B4B464',  # くすんだ黄 (未分類)
    'manual': '#96C864',     # くすんだ緑 (手動追加)
}

# BBox描画設定
BBOX_LINE_WIDTH = 2       # 枠線の太さ
BBOX_PAD = 4              # 欠陥から枠を離すpx数

# --------- 検出用モデル構築 (3クラス) ---------
def build_detection_model():
    m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES_DETECT
    )
    return m


# ==========================================
# JSON形式変換ユーティリティ
# ==========================================
def _migrate_detection(d):
    """旧形式 detection → 新形式に変換。新形式ならそのまま返す。"""
    if "raw" in d and "review" in d:
        # 新形式: そのまま (互換フィールドも念のため補完)
        d.setdefault("orig_label", d["raw"].get("label", "unknown"))
        d.setdefault("review_label", d["review"].get("label", "unlabeled"))
        d.setdefault("score", d["raw"].get("score"))
        return d

    # 旧形式 → 新形式 (処理済みデータとして扱う)
    box = d.get("box", [0, 0, 0, 0])
    score = d.get("score")
    orig_label = d.get("orig_label", "unknown")
    review_label = d.get("review_label", "unlabeled")
    confirmed = review_label != "unlabeled"

    return {
        "box": box,
        "raw": {
            "label": orig_label,
            "score": score,
            "source": "model",
        },
        "review": {
            "label": review_label,
            "confirmed": confirmed,
        },
        # 互換フィールド
        "orig_label": orig_label,
        "review_label": review_label,
        "score": score,
    }


def _make_new_detection(box, label, score, source="model"):
    """新規 detection dict を作成。"""
    return {
        "box": box,
        "raw": {
            "label": label,
            "score": score,
            "source": source,
        },
        "review": {
            "label": label if source == "model" else "unlabeled",
            "confirmed": False,
        },
        # 互換フィールド
        "orig_label": label,
        "review_label": label if source == "model" else "unlabeled",
        "score": score,
    }


def _sync_compat_fields(det):
    """review フィールドから互換フィールドを同期。"""
    det["review_label"] = det["review"]["label"]
    det["orig_label"] = det["raw"]["label"]
    det["score"] = det["raw"]["score"]


def _migrate_image_entry(filename, dets_or_entry):
    """画像単位のエントリを新形式に変換。"""
    if isinstance(dets_or_entry, dict) and "detections" in dets_or_entry:
        # 新形式 (detections + review_state)
        entry = dets_or_entry
        entry["detections"] = [_migrate_detection(d) for d in entry["detections"]]
        entry.setdefault("review_state", {"reviewed": False})
        return entry

    # 旧形式: list of detections → 処理済みと仮定して reviewed=True
    dets = dets_or_entry if isinstance(dets_or_entry, list) else []
    migrated_dets = [_migrate_detection(d) for d in dets]
    return {
        "detections": migrated_dets,
        "review_state": {"reviewed": True},
    }


def load_annotations(path):
    """annotations.json を読み込み、新形式 dict を返す。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for filename, entry in raw.items():
        result[filename] = _migrate_image_entry(filename, entry)
    return result


def save_annotations(path, all_data):
    """新形式でJSONに保存。"""
    # 各 detection の互換フィールドを同期
    for filename, entry in all_data.items():
        for det in entry["detections"]:
            _sync_compat_fields(det)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


# ==========================================
# ワーカー1: CUDA (NVIDIA GPU) プロセス
# ==========================================
def worker_cuda_process(file_subset, folder_path, model_path, result_queue):
    try:
        print(f"[CUDA] 起動開始 (対象: {len(file_subset)}枚)")
        device = torch.device("cuda")
        model = build_detection_model()
        st = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(st, strict=False)
        model.to(device)
        model.eval()
        # プロセス初期化時に1回だけ設定（プロセス単位なので正常）
        torch.backends.cudnn.benchmark = True
        print("[CUDA] 準備完了。処理開始...")

        for filename in file_subset:
            try:
                img_path = Path(folder_path) / filename
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.exif_transpose(img)
                orig_size = img.size
                img_resized = img.resize(INPUT_SIZE)
                img_tensor = TF.to_tensor(img_resized).to(device)
                with torch.no_grad():
                    out = model([img_tensor])[0]
                boxes = out['boxes'].cpu().numpy()
                scores = out['scores'].cpu().numpy()
                labels = out['labels'].cpu().numpy()
                scale_w = orig_size[0] / INPUT_SIZE[0]
                scale_h = orig_size[1] / INPUT_SIZE[1]
                if len(boxes) > 0:
                    boxes[:, [0, 2]] *= scale_w
                    boxes[:, [1, 3]] *= scale_h
                result_queue.put((filename, True, boxes, scores, labels, str(img_path)))
            except Exception as e:
                print(f"[CUDA] エラー {filename}: {e}")
                result_queue.put((filename, False, None, None, None, None))
    except Exception as e:
        print(f"[CUDA] 致命的エラー: {e}")
    finally:
        result_queue.put(None)
        print("[CUDA] 完了")


# ==========================================
# ワーカー2: OpenVINO (CPU) プロセス
# ==========================================
def worker_ov_process(file_subset, folder_path, model_path, result_queue):
    try:
        from openvino.preprocess import PrePostProcessor
        from openvino import Layout, Type
        import openvino as ov

        print(f"[OpenVINO] 起動開始 (対象: {len(file_subset)}枚)")
        core = ov.Core()
        xml_path = Path(model_path).with_name(Path(model_path).stem + "_fp16_static.xml")
        if not xml_path.exists():
            print("[OpenVINO] IRモデルが見つかりません。変換します...")
            pytorch_model = build_detection_model()
            st = torch.load(model_path, map_location="cpu", weights_only=True)
            pytorch_model.load_state_dict(st, strict=False)
            pytorch_model.eval()
            dummy = torch.randn(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
            ov_model = ov.convert_model(pytorch_model, example_input=dummy)
            ov.save_model(ov_model, xml_path, compress_to_fp16=True)
        ov_model = core.read_model(xml_path)
        ppp = PrePostProcessor(ov_model)
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NCHW')) \
            .set_color_format(ov.preprocess.ColorFormat.RGB)
        ppp.input().preprocess() \
            .convert_element_type(Type.f32) \
            .scale([255.0, 255.0, 255.0])
        ppp.input().model().set_layout(Layout('NCHW'))
        ov_model = ppp.build()
        compiled_model = core.compile_model(ov_model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        infer_queue = ov.AsyncInferQueue(compiled_model)

        def callback(infer_request, userdata):
            filename, orig_size, img_path = userdata
            out_tensors = [infer_request.get_tensor(o).data for o in compiled_model.outputs]
            boxes, scores, labels = None, None, None
            for d in out_tensors:
                if d.ndim == 2 and d.shape[1] == 4:
                    boxes = d
                elif d.ndim == 1:
                    if np.issubdtype(d.dtype, np.integer) or np.all(d == d.astype(int)):
                        labels = d
                    else:
                        scores = d
            if boxes is None: boxes = out_tensors[0]
            if scores is None: scores = out_tensors[1]
            if labels is None: labels = out_tensors[2]
            scale_w = orig_size[0] / INPUT_SIZE[0]
            scale_h = orig_size[1] / INPUT_SIZE[1]
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= scale_w
                boxes[:, [1, 3]] *= scale_h
            result_queue.put((filename, True, boxes.copy(), scores.copy(), labels.copy(), img_path))

        infer_queue.set_callback(callback)
        print("[OpenVINO] 準備完了。処理開始...")
        for filename in file_subset:
            try:
                img_path = Path(folder_path) / filename
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.exif_transpose(img)
                orig_size = img.size
                img_resized = img.resize(INPUT_SIZE)
                input_data = np.array(img_resized).transpose(2, 0, 1)
                input_data = np.expand_dims(input_data, 0)
                infer_queue.start_async({0: input_data}, (filename, orig_size, str(img_path)))
            except Exception as e:
                print(f"[OpenVINO] エラー {filename}: {e}")
        infer_queue.wait_all()
        del infer_queue
        del compiled_model
        result_queue.put(None)
        print("[OpenVINO] 完了")
    except Exception as e:
        print(f"[OpenVINO] 致命的エラー: {e}")
        result_queue.put(None)


# ==========================================
# 検出実行 (ハイブリッド)
# ==========================================
def execute_detection(model_path, folder_path, files):
    """検出を実行し、全画像の検出結果を新形式で返す。RAW_SCORE_THRESH以上を全保持。"""
    result_queue = multiprocessing.Queue()
    mid_idx = len(files) // 2
    files_cuda = files[:mid_idx]
    files_ov = files[mid_idx:]

    p_cuda = multiprocessing.Process(
        target=worker_cuda_process,
        args=(files_cuda, folder_path, model_path, result_queue)
    )
    p_ov = multiprocessing.Process(
        target=worker_ov_process,
        args=(files_ov, folder_path, model_path, result_queue)
    )

    # 全ファイル分のエントリを事前に初期化 (0検出画像も含める)
    all_data = {}
    for f in files:
        all_data[f] = {
            "detections": [],
            "review_state": {"reviewed": False},
        }

    start_time = time.time()
    p_cuda.start()
    p_ov.start()

    finished_workers = 0
    processed_count = 0

    while finished_workers < 2:
        item = result_queue.get()
        if item is None:
            finished_workers += 1
            continue
        filename, success, boxes, scores, labels, img_path = item
        if not success:
            continue

        for box, score, label in zip(boxes, scores, labels):
            if score >= RAW_SCORE_THRESH:
                lbl_id = int(label)
                lbl_name = ID2LABEL_DETECT.get(lbl_id, f"class_{lbl_id}")
                det = _make_new_detection(
                    box=[int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))],
                    label=lbl_name,
                    score=float(score),
                    source="model",
                )
                # SCORE_THRESH以上のものは初期ラベルをモデル出力に、未満はunlabeled
                if score >= SCORE_THRESH:
                    det["review"]["label"] = lbl_name
                else:
                    det["review"]["label"] = "unlabeled"
                det["review_label"] = det["review"]["label"]
                all_data[filename]["detections"].append(det)
        processed_count += 1
        if processed_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f">> 完了: {processed_count}/{len(files)} (FPS: {processed_count/elapsed:.2f})")

    p_cuda.join()
    p_ov.join()
    result_queue.close()
    result_queue.join_thread()
    total_time = time.time() - start_time
    print(f"\n検出完了: {total_time:.2f}秒, {len(all_data)}枚処理")
    return all_data


# ==========================================
# レビューGUI
# ==========================================
class ReviewGUI:
    """検出結果をレビュー・ラベル修正するGUI。"""

    DISPLAY_MAX = 900  # 表示画像の最大サイズ (px)

    def __init__(self, folder_path, all_data, annotations_save_path):
        self.folder_path = Path(folder_path)
        self.all_data = all_data  # 新形式: {filename: {detections: [...], review_state: {...}}}
        self.annotations_save_path = Path(annotations_save_path)
        self.unsaved_changes = False

        # Undo/Redo スタック
        self._undo_stack = []
        self._redo_stack = []
        self._UNDO_LIMIT = 50

        # BBox作成モード
        self.bbox_create_mode = False
        self._drag_start = None
        self._drag_rect_id = None

        # 表示閾値
        self.display_thresh = SCORE_THRESH

        # 画像キャッシュ（同一画像の再描画を高速化）
        self._img_cache_name = None
        self._img_cache_base = None

        # 500ms連打対策用
        self._last_reviewed_time = 0.0
        self._last_reviewed_filename = None
        self._last_reviewed_was_new = False

        # 検出がある画像 + 検出0個の画像も含む (全画像を対象に)
        self.filenames = sorted(all_data.keys())
        if not self.filenames:
            print("画像データが0件です。")
            return

        self.current_idx = 0
        self.selected_det_idx = None  # フィルタ後のインデックス

        self._build_gui()
        self.root.mainloop()

    # --------------------------------------------------
    # Undo / Redo
    # --------------------------------------------------
    def _push_undo(self):
        """現在の画像の状態をundoスタックにpush。"""
        filename = self.filenames[self.current_idx]
        snapshot = {
            "filename": filename,
            "detections": copy.deepcopy(self.all_data[filename]["detections"]),
            "review_state": copy.deepcopy(self.all_data[filename]["review_state"]),
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _do_undo(self):
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        filename = snapshot["filename"]
        # 現在状態をredoに退避
        redo_snapshot = {
            "filename": filename,
            "detections": copy.deepcopy(self.all_data[filename]["detections"]),
            "review_state": copy.deepcopy(self.all_data[filename]["review_state"]),
        }
        self._redo_stack.append(redo_snapshot)
        # 復元
        self.all_data[filename]["detections"] = snapshot["detections"]
        self.all_data[filename]["review_state"] = snapshot["review_state"]
        # 対象画像に移動
        if filename in self.filenames:
            self.current_idx = self.filenames.index(filename)
        self.unsaved_changes = True
        self._show_image()

    def _do_redo(self):
        if not self._redo_stack:
            return
        snapshot = self._redo_stack.pop()
        filename = snapshot["filename"]
        # 現在状態をundoに退避
        undo_snapshot = {
            "filename": filename,
            "detections": copy.deepcopy(self.all_data[filename]["detections"]),
            "review_state": copy.deepcopy(self.all_data[filename]["review_state"]),
        }
        self._undo_stack.append(undo_snapshot)
        # 復元
        self.all_data[filename]["detections"] = snapshot["detections"]
        self.all_data[filename]["review_state"] = snapshot["review_state"]
        if filename in self.filenames:
            self.current_idx = self.filenames.index(filename)
        self.unsaved_changes = True
        self._show_image()

    # --------------------------------------------------
    # フィルタ済み detection 取得
    # --------------------------------------------------
    def _visible_detections(self, filename=None):
        """display_thresh でフィルタした検出リストを (index_in_all, det) のリストで返す。"""
        if filename is None:
            filename = self.filenames[self.current_idx]
        result = []
        for i, det in enumerate(self.all_data[filename]["detections"]):
            score = det["raw"]["score"]
            if score is None:
                # 手動BBox → 常に表示
                result.append((i, det))
            elif score >= self.display_thresh:
                result.append((i, det))
        return result

    # --------------------------------------------------
    # GUI構築
    # --------------------------------------------------
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("検出結果レビュー - void / crack / other")
        self.root.configure(bg="#2b2b2b")

        # --- 上部: ナビゲーション ---
        nav_frame = tk.Frame(self.root, bg="#2b2b2b")
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_prev = tk.Button(nav_frame, text="<< 前", command=self._prev_image,
                                  width=8, bg="#444", fg="white")
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.lbl_nav = tk.Label(nav_frame, text="", bg="#2b2b2b", fg="white", font=("Arial", 11))
        self.lbl_nav.pack(side=tk.LEFT, padx=5)

        # ページジャンプ入力
        jump_frame = tk.Frame(nav_frame, bg="#2b2b2b")
        jump_frame.pack(side=tk.LEFT, padx=5)
        self.entry_jump = tk.Entry(jump_frame, width=6, font=("Arial", 11),
                                   bg="#444", fg="white", insertbackground="white",
                                   justify="center")
        self.entry_jump.pack(side=tk.LEFT)
        tk.Label(jump_frame, text=f"/{len(self.filenames)}",
                 bg="#2b2b2b", fg="#AAA", font=("Arial", 11)).pack(side=tk.LEFT)
        self.btn_jump = tk.Button(jump_frame, text="移動", command=self._jump_to_page,
                                  width=5, bg="#555", fg="white")
        self.btn_jump.pack(side=tk.LEFT, padx=3)
        self.entry_jump.bind("<Return>", lambda e: self._jump_to_page())

        self.btn_goto_unreviewed = tk.Button(nav_frame, text="次の未レビュー",
                                             command=self._goto_next_unreviewed,
                                             width=14, bg="#AA6600", fg="white")
        self.btn_goto_unreviewed.pack(side=tk.LEFT, padx=3)

        self.btn_list = tk.Button(nav_frame, text="一覧",
                                  command=self._open_list_window,
                                  width=5, bg="#555", fg="white")
        self.btn_list.pack(side=tk.LEFT, padx=3)

        # ヘルプボタン
        self.btn_help = tk.Button(nav_frame, text="？", command=self._show_help,
                                  width=3, bg="#555", fg="white", font=("Arial", 11, "bold"))
        self.btn_help.pack(side=tk.LEFT, padx=3)

        self.btn_next = tk.Button(nav_frame, text="次 >>", command=self._next_image,
                                  width=8, bg="#444", fg="white")
        self.btn_next.pack(side=tk.RIGHT, padx=3)

        # --- 閾値スライダー + BBox作成モード表示 ---
        slider_frame = tk.Frame(self.root, bg="#2b2b2b")
        slider_frame.pack(fill=tk.X, padx=5, pady=(0, 3))

        tk.Label(slider_frame, text="表示閾値:", bg="#2b2b2b", fg="#CCC",
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.thresh_var = tk.DoubleVar(value=self.display_thresh)
        self.thresh_slider = tk.Scale(
            slider_frame, from_=0.30, to=1.00, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.thresh_var,
            command=self._on_thresh_change,
            bg="#2b2b2b", fg="white", troughcolor="#444",
            highlightthickness=0, length=250, font=("Arial", 9),
        )
        self.thresh_slider.pack(side=tk.LEFT, padx=5)

        self.lbl_thresh_info = tk.Label(slider_frame, text="", bg="#2b2b2b", fg="#AAA",
                                        font=("Arial", 9))
        self.lbl_thresh_info.pack(side=tk.LEFT, padx=5)

        # BBox作成モード表示
        self.lbl_bbox_mode = tk.Label(slider_frame, text="", bg="#2b2b2b", fg="#96C864",
                                      font=("Arial", 10, "bold"))
        self.lbl_bbox_mode.pack(side=tk.RIGHT, padx=10)

        # --- 中央: 画像表示 ---
        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # キーバインド
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Up>", lambda e: self._prev_image())
        self.root.bind("<Down>", lambda e: self._next_image())
        self.root.bind("<Key-v>", lambda e: self._set_label("void"))
        self.root.bind("<Key-V>", lambda e: self._set_label("void"))
        self.root.bind("<Key-c>", lambda e: self._set_label("crack"))
        self.root.bind("<Key-C>", lambda e: self._set_label("crack"))
        self.root.bind("<Key-o>", lambda e: self._set_label("other"))
        self.root.bind("<Control-a>", lambda e: self._set_all_other())
        self.root.bind("<Key-b>", lambda e: self._toggle_bbox_mode())
        self.root.bind("<Key-B>", lambda e: self._toggle_bbox_mode())
        self.root.bind("<space>", lambda e: self._confirm_and_next())
        self.root.bind("<Return>", lambda e: self._confirm_and_next())
        self.root.bind("<Escape>", lambda e: self._on_escape())
        self.root.bind("<Key-h>", lambda e: self._show_help())
        self.root.bind("<Key-H>", lambda e: self._show_help())
        self.root.bind("<Control-z>", lambda e: self._do_undo())
        self.root.bind("<Control-Z>", lambda e: self._do_undo())
        self.root.bind("<Control-y>", lambda e: self._do_redo())
        self.root.bind("<Control-Y>", lambda e: self._do_redo())

        # ウィンドウ閉じる操作をインターセプト
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- 下部: ラベルボタン + 一括操作 ---
        ctrl_frame = tk.Frame(self.root, bg="#2b2b2b")
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_selected = tk.Label(ctrl_frame, text="BBoxを画像上でクリックして選択",
                                     bg="#2b2b2b", fg="#FFFF00", font=("Arial", 11))
        self.lbl_selected.pack(side=tk.LEFT, padx=10)

        self.btn_void = tk.Button(ctrl_frame, text="void (V)", bg="#C86464", fg="white",
                                  width=10, command=lambda: self._set_label("void"))
        self.btn_void.pack(side=tk.LEFT, padx=3)

        self.btn_crack = tk.Button(ctrl_frame, text="crack (C)", bg="#6496C8", fg="white",
                                   width=10, command=lambda: self._set_label("crack"))
        self.btn_crack.pack(side=tk.LEFT, padx=3)

        self.btn_other = tk.Button(ctrl_frame, text="other (O)", bg="#808080", fg="white",
                                   width=10, command=lambda: self._set_label("other"))
        self.btn_other.pack(side=tk.LEFT, padx=3)

        sep = tk.Label(ctrl_frame, text=" | ", bg="#2b2b2b", fg="#666")
        sep.pack(side=tk.LEFT, padx=3)

        self.btn_all_other = tk.Button(
            ctrl_frame, text="全てother (Ctrl+A)",
            bg="#AA4400", fg="white",
            width=16, command=self._set_all_other)
        self.btn_all_other.pack(side=tk.LEFT, padx=3)

        self.btn_all_void = tk.Button(ctrl_frame, text="全てvoid", bg="#C86464", fg="white",
                                      width=10, command=self._set_all_void)
        self.btn_all_void.pack(side=tk.LEFT, padx=3)

        self.btn_all_crack = tk.Button(ctrl_frame, text="全てcrack", bg="#6496C8", fg="white",
                                       width=10, command=self._set_all_crack)
        self.btn_all_crack.pack(side=tk.LEFT, padx=3)

        # --- 最下部: 保存ボタン + 進捗 ---
        bottom_frame = tk.Frame(self.root, bg="#2b2b2b")
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_save_exit = tk.Button(bottom_frame, text="保存して終了",
                                       bg="#228B22", fg="white", font=("Arial", 11, "bold"),
                                       command=self._save_and_exit)
        self.btn_save_exit.pack(side=tk.RIGHT, padx=5)

        self.btn_save = tk.Button(bottom_frame, text="中断保存 (後で再開可)",
                                  bg="#CC8800", fg="white", font=("Arial", 11, "bold"),
                                  command=self._save_intermediate)
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        self.lbl_progress = tk.Label(bottom_frame, text="", bg="#2b2b2b", fg="#AAA",
                                     font=("Arial", 10))
        self.lbl_progress.pack(side=tk.LEFT, padx=10)

        self.lbl_saved = tk.Label(bottom_frame, text="", bg="#2b2b2b", fg="#44AA44",
                                  font=("Arial", 10))
        self.lbl_saved.pack(side=tk.LEFT, padx=5)

        # --- 検出一覧パネル ---
        det_frame = tk.Frame(self.root, bg="#333")
        det_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(det_frame, text="検出一覧:", bg="#333", fg="white",
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.det_list_frame = tk.Frame(det_frame, bg="#333")
        self.det_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._show_image()

    # --------------------------------------------------
    # 閾値変更
    # --------------------------------------------------
    def _on_thresh_change(self, val):
        self.display_thresh = float(val)
        self._show_image()

    # --------------------------------------------------
    # BBox作成モード
    # --------------------------------------------------
    def _toggle_bbox_mode(self):
        self.bbox_create_mode = not self.bbox_create_mode
        if self.bbox_create_mode:
            self.lbl_bbox_mode.config(text="[BBox作成モード ON] ドラッグで矩形作成")
            self.canvas.config(cursor="crosshair")
        else:
            self.lbl_bbox_mode.config(text="")
            self.canvas.config(cursor="")
            self._cancel_drag()

    def _cancel_drag(self):
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        self._drag_start = None

    def _on_escape(self):
        if self._drag_start is not None:
            self._cancel_drag()
        elif self.bbox_create_mode:
            self._toggle_bbox_mode()

    def _on_canvas_click(self, event):
        if self.bbox_create_mode:
            self._drag_start = (event.x, event.y)
            return
        # 通常モード: BBox選択
        x, y = event.x, event.y
        for vis_idx, (all_idx, det) in enumerate(self._current_visible):
            rx1, ry1, rx2, ry2 = self.det_rects[vis_idx]
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                self._select_detection(vis_idx)
                return

    def _on_canvas_drag(self, event):
        if not self.bbox_create_mode or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        if self._drag_rect_id is not None:
            self.canvas.coords(self._drag_rect_id, x0, y0, x1, y1)
        else:
            self._drag_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#96C864", width=2, dash=(4, 4)
            )

    def _on_canvas_release(self, event):
        if not self.bbox_create_mode or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._cancel_drag()

        # 最小サイズチェック (5px)
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        # 表示座標 → 原画座標
        scale = self.display_scale
        ox1 = int(round(min(x0, x1) / scale))
        oy1 = int(round(min(y0, y1) / scale))
        ox2 = int(round(max(x0, x1) / scale))
        oy2 = int(round(max(y0, y1) / scale))

        # Undo保存 → 追加
        self._push_undo()
        filename = self.filenames[self.current_idx]
        new_det = _make_new_detection(
            box=[ox1, oy1, ox2, oy2],
            label="manual",
            score=None,
            source="human",
        )
        self.all_data[filename]["detections"].append(new_det)
        self.unsaved_changes = True

        # 再描画して新しいBBoxを選択
        self._show_image()
        # 追加したBBoxが visible の最後にあるはず
        vis = self._current_visible
        if vis:
            self._select_detection(len(vis) - 1)

    # --------------------------------------------------
    # 画像表示
    # --------------------------------------------------
    def _show_image(self):
        """現在のインデックスの画像と検出結果を表示。"""
        filename = self.filenames[self.current_idx]
        entry = self.all_data[filename]
        visible = self._visible_detections(filename)
        self._current_visible = visible

        # ナビゲーション更新
        reviewed_mark = " [済]" if entry["review_state"]["reviewed"] else ""
        self.lbl_nav.config(text=f"{filename}{reviewed_mark}")
        self.selected_det_idx = None
        self.lbl_selected.config(text="BBoxを画像上でクリックして選択")

        # ジャンプ入力欄を現在ページに同期
        self.entry_jump.delete(0, tk.END)
        self.entry_jump.insert(0, str(self.current_idx + 1))

        # 閾値情報
        total_all = len(entry["detections"])
        total_vis = len(visible)
        self.lbl_thresh_info.config(
            text=f"表示: {total_vis}/{total_all} 件 (閾値={self.display_thresh:.2f})"
        )

        # 進捗 (レビュー済み画像数)
        total_reviewed = sum(1 for f in self.filenames
                             if self.all_data[f]["review_state"]["reviewed"])
        total_unreviewed = len(self.filenames) - total_reviewed
        self.lbl_progress.config(
            text=f"レビュー済: {total_reviewed}/{len(self.filenames)}  残り: {total_unreviewed}"
        )

        # 画像読み込み（キャッシュ利用で再描画を高速化）
        img_path = self.folder_path / filename
        if (self._img_cache_name == filename
                and self._img_cache_base is not None):
            img_base = self._img_cache_base
            disp_w, disp_h = img_base.size
        else:
            try:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.exif_transpose(img)
            except Exception:
                img = Image.new(
                    "RGB", (640, 480), (30, 30, 30)
                )
            orig_w, orig_h = img.size
            scale = min(
                self.DISPLAY_MAX / orig_w,
                self.DISPLAY_MAX / orig_h,
                1.0,
            )
            disp_w = int(orig_w * scale)
            disp_h = int(orig_h * scale)
            self.display_scale = scale
            img_base = img.resize(
                (disp_w, disp_h), Image.LANCZOS
            )
            self._img_cache_name = filename
            self._img_cache_base = img_base

        img_disp = img_base.copy()
        draw = ImageDraw.Draw(img_disp)

        # BBox描画
        self.det_rects = []
        pad = BBOX_PAD
        for vis_idx, (all_idx, det) in enumerate(visible):
            bx = [int(round(c * scale)) for c in det["box"]]
            bx_draw = [bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad]
            review_label = det["review"]["label"]
            color = LABEL_COLORS.get(review_label, LABEL_COLORS["unlabeled"])

            # 手動追加で未ラベルは専用色
            if det["raw"]["source"] == "human" and review_label == "unlabeled":
                color = LABEL_COLORS["manual"]

            draw.rectangle(bx_draw, outline=color, width=BBOX_LINE_WIDTH)

            # ラベルテキスト
            score = det["raw"]["score"]
            score_str = f"{score:.2f}" if score is not None else "手動"
            label_text = f"#{vis_idx} {review_label} ({score_str})"
            txt_y = bx_draw[1] - 14
            txt_bg = [
                bx_draw[0], txt_y,
                bx_draw[0] + len(label_text) * 7,
                bx_draw[1],
            ]
            draw.rectangle(txt_bg, fill=color)
            draw.text((bx_draw[0] + 2, txt_y + 1), label_text, fill="white")
            self.det_rects.append((bx_draw[0], bx_draw[1], bx_draw[2], bx_draw[3]))

        # Canvas更新
        self.tk_img = ImageTk.PhotoImage(img_disp)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # 検出一覧ボタン更新
        for w in self.det_list_frame.winfo_children():
            w.destroy()
        for vis_idx, (all_idx, det) in enumerate(visible):
            review_label = det["review"]["label"]
            color = LABEL_COLORS.get(review_label, LABEL_COLORS["unlabeled"])
            if det["raw"]["source"] == "human" and review_label == "unlabeled":
                color = LABEL_COLORS["manual"]
            btn = tk.Button(
                self.det_list_frame,
                text=f"#{vis_idx} {review_label}",
                bg=color, fg="white", width=12,
                command=lambda idx=vis_idx: self._select_detection(idx)
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)

    # --------------------------------------------------
    # BBox選択 / ラベル操作
    # --------------------------------------------------
    def _select_detection(self, vis_idx):
        """検出を選択 (vis_idx = 表示上のインデックス)。"""
        if vis_idx >= len(self._current_visible):
            return
        all_idx, det = self._current_visible[vis_idx]
        self.selected_det_idx = vis_idx
        score = det["raw"]["score"]
        score_str = f"{score:.2f}" if score is not None else "手動"
        src = det["raw"]["source"]
        self.lbl_selected.config(
            text=f"選択中: #{vis_idx}  ラベル: {det['review']['label']}  "
                 f"スコア: {score_str}  ({src})"
        )

    def _set_label(self, label):
        """選択中の検出にラベルを設定。"""
        if self.selected_det_idx is None:
            return
        if self.selected_det_idx >= len(self._current_visible):
            return
        self._push_undo()
        all_idx, det = self._current_visible[self.selected_det_idx]
        det["review"]["label"] = label
        det["review"]["confirmed"] = True
        _sync_compat_fields(det)
        self.unsaved_changes = True
        self._show_image()

    def _set_all_other(self):
        """現在画像の表示中の全検出をotherにする。"""
        self._push_undo()
        filename = self.filenames[self.current_idx]
        for _, det in self._visible_detections(filename):
            det["review"]["label"] = "other"
            det["review"]["confirmed"] = True
            _sync_compat_fields(det)
        self.unsaved_changes = True
        self._show_image()

    def _set_all_void(self):
        self._push_undo()
        filename = self.filenames[self.current_idx]
        for _, det in self._visible_detections(filename):
            det["review"]["label"] = "void"
            det["review"]["confirmed"] = True
            _sync_compat_fields(det)
        self.unsaved_changes = True
        self._show_image()

    def _set_all_crack(self):
        self._push_undo()
        filename = self.filenames[self.current_idx]
        for _, det in self._visible_detections(filename):
            det["review"]["label"] = "crack"
            det["review"]["confirmed"] = True
            _sync_compat_fields(det)
        self.unsaved_changes = True
        self._show_image()

    # --------------------------------------------------
    # 未確定手動BBox自動削除 (画像を離れる前)
    # --------------------------------------------------
    def _purge_unconfirmed_manual(self):
        """現在画像の未確定手動BBoxを削除。削除があればTrue。"""
        filename = self.filenames[self.current_idx]
        dets = self.all_data[filename]["detections"]
        new_dets = [d for d in dets if not (
            d["raw"]["source"] == "human" and not d["review"]["confirmed"]
        )]
        if len(new_dets) < len(dets):
            self.all_data[filename]["detections"] = new_dets
            return True
        return False

    # --------------------------------------------------
    # reviewed 確定 (500ms連打対策付き)
    # --------------------------------------------------
    def _mark_reviewed(self):
        """現在画像を reviewed=True にする。500ms連打対策で前画像を取り消す可能性あり。"""
        now = time.time()
        filename = self.filenames[self.current_idx]

        # 500ms以内の連打: 前回「新規にreviewedにした」画像だけ取り消す
        if (now - self._last_reviewed_time < 0.5
                and self._last_reviewed_was_new
                and self._last_reviewed_filename is not None
                and self._last_reviewed_filename in self.all_data):
            self.all_data[self._last_reviewed_filename]["review_state"]["reviewed"] = False

        # 現在画像のreviewed状態を記録してからセット
        was_already = self.all_data[filename]["review_state"]["reviewed"]
        self.all_data[filename]["review_state"]["reviewed"] = True

        self._last_reviewed_time = now
        self._last_reviewed_filename = filename
        self._last_reviewed_was_new = not was_already

    # --------------------------------------------------
    # 共通ナビゲーション
    # --------------------------------------------------
    def _navigate(self, target_idx, mark_reviewed=True):
        """画像を移動する共通関数。移動前に未確定BBox削除とreviewed処理を行う。"""
        if target_idx < 0 or target_idx >= len(self.filenames):
            return
        if target_idx == self.current_idx and not mark_reviewed:
            return

        # Undo保存 (未確定BBox削除 + reviewed変更を一括で戻せるように)
        self._push_undo()

        # 未確定手動BBox削除
        self._purge_unconfirmed_manual()

        # reviewed 確定
        if mark_reviewed:
            self._mark_reviewed()

        self.unsaved_changes = True
        self.current_idx = target_idx
        self._show_image()

    def _confirm_and_next(self):
        """Space/Enter: reviewed=true にして次へ。"""
        next_idx = min(self.current_idx + 1, len(self.filenames) - 1)
        self._navigate(next_idx, mark_reviewed=True)

    # --------------------------------------------------
    # reviewed 判定
    # --------------------------------------------------
    def _is_reviewed(self, filename):
        return self.all_data[filename]["review_state"]["reviewed"]

    def _prev_image(self):
        if self.current_idx > 0:
            self._navigate(self.current_idx - 1, mark_reviewed=True)

    def _next_image(self):
        if self.current_idx < len(self.filenames) - 1:
            self._navigate(self.current_idx + 1, mark_reviewed=True)

    def _jump_to_page(self):
        try:
            page = int(self.entry_jump.get().strip())
            page = max(1, min(page, len(self.filenames)))
            self._navigate(page - 1, mark_reviewed=False)
        except ValueError:
            pass

    def _goto_next_unreviewed(self):
        total = len(self.filenames)
        for offset in range(1, total + 1):
            idx = (self.current_idx + offset) % total
            if not self._is_reviewed(self.filenames[idx]):
                self._navigate(idx, mark_reviewed=False)
                return
        messagebox.showinfo("完了", "未レビューの画像はありません。")

    # --------------------------------------------------
    # ヘルプ
    # --------------------------------------------------
    def _show_help(self):
        help_text = (
            "=== キーボードショートカット ===\n\n"
            "矢印キー (← → ↑ ↓)    画像をレビュー済みにして前後移動\n"
            "Space / Enter              画像をレビュー済みにして次へ\n"
            "  ※ 500ms以内の連打は飛ばし扱い (未レビューに戻る)\n"
            "  ※ 既にレビュー済みの画像は連打しても未レビュー化しない\n\n"
            "V                          選択BBoxを void に\n"
            "C                          選択BBoxを crack に\n"
            "O                          選択BBoxを other に\n"
            "Ctrl+A                     全BBoxを other に (一括)\n\n"
            "B                          BBox作成モード ON/OFF\n"
            "  作成モード中: Canvasドラッグで矩形作成\n"
            "Esc                        作成中キャンセル / モードOFF\n\n"
            "Ctrl+Z                     元に戻す (Undo)\n"
            "Ctrl+Y                     やり直し (Redo)\n\n"
            "H                          このヘルプを表示\n\n"
            "=== 自動動作 ===\n"
            "画像移動時、未確定の手動BBox (V/C/Oが未設定) は自動削除されます。\n"
            "(Ctrl+Zで戻せます)\n\n"
            "=== 表示閾値スライダー ===\n"
            "スライダーで表示するスコア閾値を変更できます。\n"
            "表示フィルタのみで、データ自体は変更されません。\n"
            "手動BBox (score=null) は閾値に関係なく常に表示されます。\n"
            "学習時の条件とは無関係です。"
        )
        messagebox.showinfo("ヘルプ - ショートカット一覧", help_text)

    # --------------------------------------------------
    # 一覧ウィンドウ
    # --------------------------------------------------
    def _open_list_window(self):
        list_win = tk.Toplevel(self.root)
        list_win.title("画像一覧 - クリックで移動")
        list_win.geometry("500x600")
        list_win.configure(bg="#2b2b2b")

        filter_frame = tk.Frame(list_win, bg="#2b2b2b")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        self._list_filter = tk.StringVar(value="all")
        for val, text in [("all", "全て"), ("unreviewed", "未レビューのみ"), ("reviewed", "レビュー済のみ")]:
            tk.Radiobutton(filter_frame, text=text, variable=self._list_filter, value=val,
                           bg="#2b2b2b", fg="white", selectcolor="#444",
                           activebackground="#2b2b2b", activeforeground="white",
                           command=lambda w=list_win: self._refresh_list(w)).pack(side=tk.LEFT, padx=5)

        container = tk.Frame(list_win, bg="#2b2b2b")
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._list_canvas = tk.Canvas(container, bg="#1e1e1e", yscrollcommand=scrollbar.set,
                                      highlightthickness=0)
        self._list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._list_canvas.yview)

        self._list_inner = tk.Frame(self._list_canvas, bg="#1e1e1e")
        self._list_canvas.create_window((0, 0), window=self._list_inner, anchor="nw")
        self._list_inner.bind("<Configure>",
                              lambda e: self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all")))
        self._list_canvas.bind_all("<MouseWheel>",
                                   lambda e: self._list_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._list_window = list_win
        self._refresh_list(list_win)

        def on_list_close():
            try:
                self._list_canvas.unbind_all("<MouseWheel>")
            except Exception:
                pass
            list_win.destroy()
        list_win.protocol("WM_DELETE_WINDOW", on_list_close)

    def _refresh_list(self, list_win):
        for w in self._list_inner.winfo_children():
            w.destroy()

        filt = self._list_filter.get()
        count_shown = 0

        for idx, filename in enumerate(self.filenames):
            reviewed = self._is_reviewed(filename)
            if filt == "unreviewed" and reviewed:
                continue
            if filt == "reviewed" and not reviewed:
                continue

            dets = self.all_data[filename]["detections"]
            n_total = len(dets)
            n_other = sum(1 for d in dets if d["review"]["label"] == "other")
            n_void = sum(1 for d in dets if d["review"]["label"] == "void")
            n_crack = sum(1 for d in dets if d["review"]["label"] == "crack")

            if reviewed:
                bg_color = "#1a3a1a"
                status = "済"
            else:
                bg_color = "#3a1a1a"
                status = "未"

            if idx == self.current_idx:
                bg_color = "#3a3a00"

            row = tk.Frame(self._list_inner, bg=bg_color, cursor="hand2")
            row.pack(fill=tk.X, padx=2, pady=1)

            lbl = tk.Label(
                row,
                text=f"[{status}] {idx+1:4d}  {filename}    "
                     f"V:{n_void} C:{n_crack} O:{n_other} (計{n_total})",
                bg=bg_color, fg="white", font=("Consolas", 9),
                anchor="w", padx=5, pady=2
            )
            lbl.pack(fill=tk.X)

            for widget in (row, lbl):
                widget.bind("<Button-1>", lambda e, i=idx: self._jump_from_list(i))

            count_shown += 1

        total_reviewed = sum(1 for f in self.filenames if self._is_reviewed(f))
        list_win.title(f"画像一覧 ({count_shown}件表示) - レビュー済: {total_reviewed}/{len(self.filenames)}")

    def _jump_from_list(self, idx):
        self._navigate(idx, mark_reviewed=False)
        if hasattr(self, '_list_window') and self._list_window.winfo_exists():
            self._refresh_list(self._list_window)

    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    def _do_save(self):
        save_annotations(self.annotations_save_path, self.all_data)
        self.unsaved_changes = False

        count_void = 0
        count_crack = 0
        count_other = 0
        
        # CSV出力の準備
        csv_path = self.folder_path / "review_summary.csv"
        csv_rows = []
        
        for filename, entry in self.all_data.items():
            file_void = 0
            file_crack = 0
            boxes_str_list = []
            for d in entry["detections"]:
                rl = d["review"]["label"]
                if rl == "void":
                    count_void += 1
                    file_void += 1
                    boxes_str_list.append(f"void:{d['box']}")
                elif rl == "crack":
                    count_crack += 1
                    file_crack += 1
                    boxes_str_list.append(f"crack:{d['box']}")
                elif rl == "other":
                    count_other += 1

            file_total = file_void + file_crack
            boxes_str = ";".join(boxes_str_list)
            csv_rows.append([filename, file_void, file_crack, file_total, boxes_str])

        # CSVに書き込み
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "void_count", "crack_count", "total_count", "boxes_info"])
                writer.writerows(csv_rows)
            print(f"\nレビューサマリーCSVを保存しました: {csv_path}")
        except Exception as e:
            print(f"\nCSVの保存に失敗しました: {e}")

        print(f"\nアノテーション保存完了: {self.annotations_save_path}")
        print(f"  void: {count_void}, crack: {count_crack}, other: {count_other}")
        return count_void, count_crack, count_other

    def _save_intermediate(self):
        v, c, o = self._do_save()
        self.lbl_saved.config(
            text=f"保存済 (void:{v} crack:{c} other:{o}) - モード2で再開可能"
        )

    def _save_and_exit(self):
        if not messagebox.askyesno("確認", "アノテーションを保存して終了しますか?"):
            return
        self._do_save()
        self.root.destroy()

    def _on_close(self):
        if self.unsaved_changes:
            result = messagebox.askyesnocancel(
                "未保存の変更があります",
                "保存してから終了しますか?\n\n"
                "「はい」→ 保存して終了\n"
                "「いいえ」→ 保存せず終了\n"
                "「キャンセル」→ 戻る"
            )
            if result is None:
                return
            if result:
                self._do_save()
        self.root.destroy()


# ==========================================
# ファインチューニング用Dataset
# ==========================================
class FineTuneDataset(Dataset):
    """
    アノテーションJSONから学習用データセットを構築。
    新形式 (detections + review_state) / 旧形式 両方に対応。
    - review.label == "void"  → ラベルID 1
    - review.label == "crack" → ラベルID 2
    - review.label == "other" / "unlabeled" → boxesに含めない
    - boxes が空の画像 (=hard negative) も学習に含める (背景学習)
    - reviewed==True の画像のみ学習対象 (新形式の場合)
    """

    def __init__(self, annotations_path, image_dir, input_size=INPUT_SIZE, augment=True):
        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.augment = augment

        with open(annotations_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        for filename, entry in raw.items():
            img_path = self.image_dir / filename
            if not img_path.exists():
                continue

            # 新旧形式対応
            if isinstance(entry, dict) and "detections" in entry:
                dets = entry["detections"]
                # 新形式: reviewed==True の画像のみ学習対象
                reviewed = entry.get("review_state", {}).get("reviewed", False)
                if not reviewed:
                    continue
            elif isinstance(entry, list):
                dets = entry
                # 旧形式: 全画像を学習対象とする
            else:
                continue

            boxes = []
            labels = []
            for d in dets:
                # review.label (新) or review_label (旧) を取得
                if "review" in d and isinstance(d["review"], dict):
                    rl = d["review"].get("label", "unlabeled")
                    confirmed = d["review"].get("confirmed", False)
                else:
                    rl = d.get("review_label", "unlabeled")
                    confirmed = rl != "unlabeled"

                # 正例: confirmed かつ void/crack のみ
                if confirmed and rl == "void":
                    boxes.append(d["box"])
                    labels.append(1)
                elif confirmed and rl == "crack":
                    boxes.append(d["box"])
                    labels.append(2)
                # other/unlabeled/未confirmed → boxesに入れない (hard negative)

            # boxes が空でも学習に含める (背景画像として機能)
            self.samples.append({
                "filename": filename,
                "img_path": str(img_path),
                "boxes": boxes,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        img = ImageOps.exif_transpose(img)
        orig_w, orig_h = img.size

        img = img.resize(self.input_size)
        scale_w = self.input_size[0] / orig_w
        scale_h = self.input_size[1] / orig_h

        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip_h = True
        else:
            flip_h = False

        img_tensor = TF.to_tensor(img)

        boxes = []
        for b in sample["boxes"]:
            x1 = b[0] * scale_w
            y1 = b[1] * scale_h
            x2 = b[2] * scale_w
            y2 = b[3] * scale_h
            if flip_h:
                x1, x2 = self.input_size[0] - x2, self.input_size[0] - x1
            boxes.append([x1, y1, x2, y2])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(sample["labels"], dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }
        return img_tensor, target


def collate_batch(batch):
    return tuple(zip(*batch))


# ==========================================
# ファインチューニング実行
# ==========================================
class FinetuneProgressGUI:
    """ファインチューニングの進捗をGUIで表示。早期停止機能付き。"""

    def __init__(self, total_epochs, total_batches_per_epoch):
        self.total_epochs = total_epochs
        self.total_batches = total_batches_per_epoch
        self.total_steps = total_epochs * total_batches_per_epoch

        self.stop_requested = False
        self.cancel_requested = False

        self.root = tk.Tk()
        self.root.title("ファインチューニング進捗")
        self.root.geometry("560x400")
        self.root.configure(bg="#2b2b2b")
        self.root.resizable(False, False)

        self.lbl_title = tk.Label(self.root, text="ファインチューニング実行中",
                                  bg="#2b2b2b", fg="white", font=("Arial", 14, "bold"))
        self.lbl_title.pack(pady=(15, 10))

        frame_epoch = tk.Frame(self.root, bg="#2b2b2b")
        frame_epoch.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame_epoch, text="Epoch:", bg="#2b2b2b", fg="#CCC",
                 font=("Arial", 10), width=8, anchor="w").pack(side=tk.LEFT)
        self.bar_epoch = ttk.Progressbar(frame_epoch, length=350, mode="determinate",
                                         maximum=total_epochs)
        self.bar_epoch.pack(side=tk.LEFT, padx=5)
        self.lbl_epoch = tk.Label(frame_epoch, text=f"0/{total_epochs}",
                                  bg="#2b2b2b", fg="white", font=("Arial", 10), width=10)
        self.lbl_epoch.pack(side=tk.LEFT)

        frame_batch = tk.Frame(self.root, bg="#2b2b2b")
        frame_batch.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame_batch, text="Batch:", bg="#2b2b2b", fg="#CCC",
                 font=("Arial", 10), width=8, anchor="w").pack(side=tk.LEFT)
        self.bar_batch = ttk.Progressbar(frame_batch, length=350, mode="determinate",
                                         maximum=max(total_batches_per_epoch, 1))
        self.bar_batch.pack(side=tk.LEFT, padx=5)
        self.lbl_batch = tk.Label(frame_batch, text=f"0/{total_batches_per_epoch}",
                                  bg="#2b2b2b", fg="white", font=("Arial", 10), width=10)
        self.lbl_batch.pack(side=tk.LEFT)

        frame_total = tk.Frame(self.root, bg="#2b2b2b")
        frame_total.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame_total, text="全体:", bg="#2b2b2b", fg="#CCC",
                 font=("Arial", 10), width=8, anchor="w").pack(side=tk.LEFT)
        self.bar_total = ttk.Progressbar(frame_total, length=350, mode="determinate",
                                         maximum=max(self.total_steps, 1))
        self.bar_total.pack(side=tk.LEFT, padx=5)
        self.lbl_percent = tk.Label(frame_total, text="0%",
                                    bg="#2b2b2b", fg="white", font=("Arial", 10), width=10)
        self.lbl_percent.pack(side=tk.LEFT)

        info_frame = tk.Frame(self.root, bg="#333")
        info_frame.pack(fill=tk.X, padx=20, pady=(15, 5))
        self.lbl_loss = tk.Label(info_frame, text="Loss: ---  |  Best: ---",
                                 bg="#333", fg="#FFD700", font=("Arial", 11))
        self.lbl_loss.pack(pady=5)

        time_frame = tk.Frame(self.root, bg="#2b2b2b")
        time_frame.pack(fill=tk.X, padx=20, pady=5)
        self.lbl_elapsed = tk.Label(time_frame, text="経過: 00:00",
                                    bg="#2b2b2b", fg="#AAA", font=("Arial", 10))
        self.lbl_elapsed.pack(side=tk.LEFT)
        self.lbl_eta = tk.Label(time_frame, text="残り: 計算中...",
                                bg="#2b2b2b", fg="#AAA", font=("Arial", 10))
        self.lbl_eta.pack(side=tk.RIGHT)
        self.lbl_speed = tk.Label(time_frame, text="",
                                  bg="#2b2b2b", fg="#888", font=("Arial", 9))
        self.lbl_speed.pack()

        stop_frame = tk.Frame(self.root, bg="#2b2b2b")
        stop_frame.pack(fill=tk.X, padx=20, pady=(10, 5))
        self.lbl_stop_status = tk.Label(stop_frame, text="",
                                        bg="#2b2b2b", fg="#FF8800", font=("Arial", 10))
        self.lbl_stop_status.pack(side=tk.LEFT, padx=5)
        self.btn_cancel = tk.Button(stop_frame, text="中止 (保存しない)",
                                    bg="#882222", fg="white", font=("Arial", 10),
                                    command=self._on_cancel)
        self.btn_cancel.pack(side=tk.RIGHT, padx=5)
        self.btn_stop_save = tk.Button(stop_frame, text="ここで停止して保存",
                                       bg="#CC6600", fg="white", font=("Arial", 10, "bold"),
                                       command=self._on_stop_save)
        self.btn_stop_save.pack(side=tk.RIGHT, padx=5)

        self.start_time = time.time()
        self.current_step = 0
        self.finished = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def update_totals(self, total_epochs, total_batches):
        """epoch数/batch数が変更された時にプログレスバーを更新。"""
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.total_steps = total_epochs * total_batches
        self.bar_epoch.config(maximum=max(total_epochs, 1))
        self.bar_batch.config(maximum=max(total_batches, 1))
        self.bar_total.config(maximum=max(self.total_steps, 1))

    def _on_stop_save(self):
        if messagebox.askyesno("確認", "現時点のベストモデルを保存して学習を停止しますか?"):
            self.stop_requested = True
            self.btn_stop_save.config(state=tk.DISABLED, text="停止中...")
            self.btn_cancel.config(state=tk.DISABLED)
            self.lbl_stop_status.config(text="停止要求受付済み (現在のバッチ完了後に停止します)")

    def _on_cancel(self):
        if messagebox.askyesno("確認", "学習を中止しますか?\nモデルは保存されません。"):
            self.cancel_requested = True
            self.btn_stop_save.config(state=tk.DISABLED)
            self.btn_cancel.config(state=tk.DISABLED, text="中止中...")
            self.lbl_stop_status.config(text="中止要求受付済み...")

    def _on_window_close(self):
        result = messagebox.askyesnocancel(
            "学習中です",
            "ベストモデルを保存して停止しますか?\n\n"
            "「はい」→ 保存して停止\n"
            "「いいえ」→ 保存せず中止\n"
            "「キャンセル」→ 学習を続ける"
        )
        if result is None:
            return
        if result:
            self.stop_requested = True
        else:
            self.cancel_requested = True
        self.btn_stop_save.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)
        self.lbl_stop_status.config(text="停止要求受付済み...")

    @staticmethod
    def _fmt_time(seconds):
        seconds = int(seconds)
        if seconds < 3600:
            return f"{seconds // 60:02d}:{seconds % 60:02d}"
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:d}:{m:02d}:{s:02d}"

    def update(self, epoch, batch, avg_loss, best_loss):
        self.current_step = epoch * self.total_batches + batch + 1
        elapsed = time.time() - self.start_time
        self.bar_epoch["value"] = epoch + 1
        self.lbl_epoch.config(text=f"{epoch + 1}/{self.total_epochs}")
        self.bar_batch["value"] = batch + 1
        self.lbl_batch.config(text=f"{batch + 1}/{self.total_batches}")
        self.bar_total["value"] = self.current_step
        pct = (self.current_step / max(self.total_steps, 1)) * 100
        self.lbl_percent.config(text=f"{pct:.0f}%")
        self.lbl_loss.config(text=f"Loss: {avg_loss:.4f}  |  Best: {best_loss:.4f}")
        self.lbl_elapsed.config(text=f"経過: {self._fmt_time(elapsed)}")
        if self.current_step > 0:
            sec_per_step = elapsed / self.current_step
            remaining = sec_per_step * (self.total_steps - self.current_step)
            self.lbl_eta.config(text=f"残り: {self._fmt_time(remaining)}")
            self.lbl_speed.config(text=f"({sec_per_step:.2f} 秒/batch)")

    def finish(self, output_path, best_loss, early_stopped=False):
        self.finished = True
        elapsed = time.time() - self.start_time
        if early_stopped:
            self.lbl_title.config(text="早期停止 - ベストモデル保存済み", fg="#FF8800")
            self.lbl_loss.config(text=f"早期停止! Best Loss: {best_loss:.4f}")
        else:
            self.lbl_title.config(text="ファインチューニング完了")
            self.bar_epoch["value"] = self.total_epochs
            self.bar_batch["value"] = self.total_batches
            self.bar_total["value"] = self.total_steps
            self.lbl_percent.config(text="100%")
            self.lbl_loss.config(text=f"完了! Best Loss: {best_loss:.4f}")
        self.lbl_eta.config(text="")
        self.lbl_elapsed.config(text=f"合計: {self._fmt_time(elapsed)}")
        self.lbl_speed.config(text=f"保存先: {Path(output_path).name}")
        self.lbl_stop_status.config(text="")
        self.btn_stop_save.pack_forget()
        self.btn_cancel.pack_forget()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        tk.Button(self.root, text="閉じる", bg="#228B22", fg="white",
                  font=("Arial", 12, "bold"), width=15,
                  command=self.root.destroy).pack(pady=10)

    def finish_cancelled(self):
        self.finished = True
        self.lbl_title.config(text="学習中止", fg="#FF4444")
        self.lbl_loss.config(text="中止されました (モデル未保存)")
        self.lbl_eta.config(text="")
        self.lbl_stop_status.config(text="")
        self.btn_stop_save.pack_forget()
        self.btn_cancel.pack_forget()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        tk.Button(self.root, text="閉じる", bg="#882222", fg="white",
                  font=("Arial", 12, "bold"), width=15,
                  command=self.root.destroy).pack(pady=10)


# ==========================================
# 学習設定管理
# ==========================================
class TrainConfig:
    """学習中の設定を管理。applied(適用中) と pending(予約) を分離。"""
    BATCH_KEYS = {"input_size", "freeze_mode"}
    EPOCH_KEYS = {"num_workers", "batch_size", "lr", "epochs"}

    def __init__(self, input_size=640, num_workers=0, batch_size=2,
                 epochs=15, lr=0.0005, freeze_mode="all"):
        self.applied = {
            "input_size": input_size,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "freeze_mode": freeze_mode,
        }
        self.pending = {}

    def set_pending(self, key, value):
        """予約値を設定。適用中と同じなら予約をクリア。"""
        if value == self.applied[key]:
            self.pending.pop(key, None)
        else:
            self.pending[key] = value

    def apply_batch_level(self):
        """batch-level の pending を適用。適用されたキーの set を返す。"""
        applied = set()
        for key in list(self.pending):
            if key in self.BATCH_KEYS:
                self.applied[key] = self.pending.pop(key)
                applied.add(key)
        return applied

    def apply_epoch_level(self):
        """epoch-level の pending を適用。適用されたキーの set を返す。"""
        applied = set()
        for key in list(self.pending):
            if key in self.EPOCH_KEYS:
                self.applied[key] = self.pending.pop(key)
                applied.add(key)
        return applied


def _apply_freeze_mode(model, mode):
    """モデルの freeze 状態を切り替える。"""
    for name, param in model.named_parameters():
        if mode == "head":
            param.requires_grad = "roi_heads" in name
        else:  # "all"
            param.requires_grad = True


def _build_optimizer(model, lr, freeze_mode):
    """freeze_mode に応じた optimizer を構築。"""
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if freeze_mode == "all" and "backbone" in name:
            params.append({"params": param, "lr": lr * 0.1})
        else:
            params.append({"params": param, "lr": lr})
    return torch.optim.SGD(params, momentum=0.9, weight_decay=0.0005)


# ==========================================
# 学習設定コントロールGUI
# ==========================================
class TrainingControlGUI:
    """学習中に設定を変更するためのコントロールウィンドウ。"""

    TIMING = {
        "input_size": "次batch",
        "freeze_mode": "次batch (optimizer: 次epoch)",
        "num_workers": "次epoch",
        "batch_size": "次epoch",
        "lr": "次epoch",
        "epochs": "次epoch",
    }

    def __init__(self, parent, config):
        self.config = config
        self.win = tk.Toplevel(parent)
        self.win.title("学習設定コントロール")
        self.win.geometry("500x480")
        self.win.configure(bg="#2b2b2b")
        self.win.resizable(False, False)

        self._status_labels = {}
        self._build_ui()
        self._refresh()

    def _alive(self):
        try:
            return self.win.winfo_exists()
        except Exception:
            return False

    def _build_ui(self):
        tk.Label(self.win, text="学習設定コントロール", bg="#2b2b2b", fg="white",
                 font=("Arial", 12, "bold")).pack(pady=(10, 5))

        sf = tk.Frame(self.win, bg="#2b2b2b")
        sf.pack(fill=tk.X, padx=10, pady=5)
        lo = {"bg": "#2b2b2b", "fg": "#CCC", "font": ("Arial", 10), "anchor": "w"}
        so = {"bg": "#2b2b2b", "fg": "#88FF88", "font": ("Arial", 9), "anchor": "w"}
        row = 0

        # input_size
        tk.Label(sf, text="input_size:", width=12, **lo).grid(row=row, column=0, sticky="w", pady=3)
        self._isz = tk.IntVar(value=self.config.applied["input_size"])
        spn = tk.Spinbox(sf, values=(384, 512, 640), textvariable=self._isz,
                         width=6, bg="#444", fg="white", font=("Arial", 10),
                         command=self._on_isz)
        spn.grid(row=row, column=1, padx=5, pady=3)
        spn.bind("<Return>", lambda e: self._on_isz())
        self._status_labels["input_size"] = tk.Label(sf, width=38, **so)
        self._status_labels["input_size"].grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # freeze_mode
        tk.Label(sf, text="freeze:", width=12, **lo).grid(row=row, column=0, sticky="w", pady=3)
        ff = tk.Frame(sf, bg="#2b2b2b")
        ff.grid(row=row, column=1, sticky="w", pady=3)
        self._frz = tk.StringVar(value=self.config.applied["freeze_mode"])
        for val, txt in [("all", "all"), ("head", "head")]:
            tk.Radiobutton(ff, text=txt, variable=self._frz, value=val,
                           bg="#2b2b2b", fg="white", selectcolor="#444",
                           activebackground="#2b2b2b", activeforeground="white",
                           command=self._on_frz).pack(side=tk.LEFT, padx=2)
        self._status_labels["freeze_mode"] = tk.Label(sf, width=38, **so)
        self._status_labels["freeze_mode"].grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # batch_size
        tk.Label(sf, text="batch_size:", width=12, **lo).grid(row=row, column=0, sticky="w", pady=3)
        self._bs = tk.IntVar(value=self.config.applied["batch_size"])
        spn = tk.Spinbox(sf, from_=1, to=4, textvariable=self._bs,
                         width=6, bg="#444", fg="white", font=("Arial", 10),
                         command=self._on_bs)
        spn.grid(row=row, column=1, padx=5, pady=3)
        spn.bind("<Return>", lambda e: self._on_bs())
        self._status_labels["batch_size"] = tk.Label(sf, width=38, **so)
        self._status_labels["batch_size"].grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # num_workers
        tk.Label(sf, text="num_workers:", width=12, **lo).grid(row=row, column=0, sticky="w", pady=3)
        self._nw = tk.IntVar(value=self.config.applied["num_workers"])
        spn = tk.Spinbox(sf, from_=0, to=8, textvariable=self._nw,
                         width=6, bg="#444", fg="white", font=("Arial", 10),
                         command=self._on_nw)
        spn.grid(row=row, column=1, padx=5, pady=3)
        spn.bind("<Return>", lambda e: self._on_nw())
        self._status_labels["num_workers"] = tk.Label(sf, width=38, **so)
        self._status_labels["num_workers"].grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # lr
        tk.Label(sf, text="lr:", width=12, **lo).grid(row=row, column=0, sticky="w", pady=3)
        self._lr = tk.StringVar(value=str(self.config.applied["lr"]))
        ent = tk.Entry(sf, textvariable=self._lr, width=10,
                       bg="#444", fg="white", font=("Arial", 10), insertbackground="white")
        ent.grid(row=row, column=1, padx=5, pady=3, sticky="w")
        ent.bind("<Return>", lambda e: self._on_lr())
        ent.bind("<FocusOut>", lambda e: self._on_lr())
        self._status_labels["lr"] = tk.Label(sf, width=38, **so)
        self._status_labels["lr"].grid(row=row, column=2, sticky="w", pady=3)
        row += 1

        # epochs
        tk.Label(sf, text="epochs:", width=12, **lo).grid(row=row, column=0, sticky="w", pady=3)
        self._ep = tk.IntVar(value=self.config.applied["epochs"])
        spn = tk.Spinbox(sf, from_=1, to=50, textvariable=self._ep,
                         width=6, bg="#444", fg="white", font=("Arial", 10),
                         command=self._on_ep)
        spn.grid(row=row, column=1, padx=5, pady=3)
        spn.bind("<Return>", lambda e: self._on_ep())
        self._status_labels["epochs"] = tk.Label(sf, width=38, **so)
        self._status_labels["epochs"].grid(row=row, column=2, sticky="w", pady=3)

        # Separator + Log
        tk.Frame(self.win, bg="#555", height=1).pack(fill=tk.X, padx=10, pady=8)
        tk.Label(self.win, text="適用ログ:", bg="#2b2b2b", fg="#AAA",
                 font=("Arial", 9)).pack(anchor="w", padx=12)
        lf = tk.Frame(self.win, bg="#1e1e1e")
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))
        self._log = tk.Text(lf, height=8, bg="#1e1e1e", fg="#CCC",
                            font=("Consolas", 9), state=tk.DISABLED, wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True)

    # --- コールバック ---
    def _on_isz(self):
        try:
            v = self._isz.get()
            if v in (384, 512, 640):
                self.config.set_pending("input_size", v)
                self._refresh()
        except (tk.TclError, ValueError):
            pass

    def _on_frz(self):
        self.config.set_pending("freeze_mode", self._frz.get())
        self._refresh()

    def _on_bs(self):
        try:
            v = self._bs.get()
            if 1 <= v <= 4:
                self.config.set_pending("batch_size", v)
                self._refresh()
        except (tk.TclError, ValueError):
            pass

    def _on_nw(self):
        try:
            v = self._nw.get()
            if 0 <= v <= 8:
                self.config.set_pending("num_workers", v)
                self._refresh()
        except (tk.TclError, ValueError):
            pass

    def _on_lr(self):
        try:
            v = float(self._lr.get())
            if v > 0:
                self.config.set_pending("lr", v)
                self._refresh()
        except (ValueError, tk.TclError):
            pass

    def _on_ep(self):
        try:
            v = self._ep.get()
            if 1 <= v <= 50:
                self.config.set_pending("epochs", v)
                self._refresh()
        except (tk.TclError, ValueError):
            pass

    # --- 表示更新 ---
    def _refresh(self):
        if not self._alive():
            return
        for key, lbl in self._status_labels.items():
            a = self.config.applied[key]
            if key in self.config.pending:
                p = self.config.pending[key]
                t = self.TIMING[key]
                lbl.config(text=f"適用中: {a}  →  {p} ({t})", fg="#FFD700")
            else:
                lbl.config(text=f"適用中: {a}", fg="#88FF88")

    def refresh(self):
        """外部から表示更新。"""
        self._refresh()

    def add_log(self, epoch, batch, msg):
        if not self._alive():
            return
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, f"[E{epoch + 1} B{batch + 1}] {msg}\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)


def _train_worker(model, dataset, device, config, msg_queue, stop_check_fn,
                  output_model_path):
    """学習ループ (別スレッドで実行)。進捗やログはmsg_queueで送信。"""
    try:
        loader = DataLoader(dataset, batch_size=config.applied["batch_size"],
                            shuffle=True, collate_fn=collate_batch,
                            num_workers=config.applied["num_workers"])
        total_batches = len(loader)
        msg_queue.put(("update_totals", config.applied["epochs"], total_batches))

        _apply_freeze_mode(model, config.applied["freeze_mode"])
        optimizer = _build_optimizer(model, config.applied["lr"],
                                     config.applied["freeze_mode"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.applied["epochs"])

        best_loss = float("inf")
        best_state = None
        stopped_early = False
        cancelled = False
        pending_optimizer_rebuild = False

        current_epoch = 0
        while current_epoch < config.applied["epochs"]:
            # === Epoch開始: epoch-level pending を適用 ===
            epoch_applied = config.apply_epoch_level()
            need_loader_rebuild = bool({"batch_size", "num_workers"} & epoch_applied)
            need_optimizer_rebuild = pending_optimizer_rebuild
            pending_optimizer_rebuild = False

            if "lr" in epoch_applied:
                need_optimizer_rebuild = True
                msg_queue.put(("log", current_epoch, 0, f"lr: →{config.applied['lr']}"))
            if "epochs" in epoch_applied:
                msg_queue.put(("log", current_epoch, 0,
                               f"epochs: →{config.applied['epochs']}"))

            if need_loader_rebuild:
                loader = DataLoader(dataset, batch_size=config.applied["batch_size"],
                                    shuffle=True, collate_fn=collate_batch,
                                    num_workers=config.applied["num_workers"])
                total_batches = len(loader)
                for key in ("batch_size", "num_workers"):
                    if key in epoch_applied:
                        msg_queue.put(("log", current_epoch, 0,
                                       f"{key}: →{config.applied[key]}"))

            if need_optimizer_rebuild:
                _apply_freeze_mode(model, config.applied["freeze_mode"])
                optimizer = _build_optimizer(model, config.applied["lr"],
                                             config.applied["freeze_mode"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(config.applied["epochs"] - current_epoch, 1))
                msg_queue.put(("log", current_epoch, 0, "optimizer 再構築"))

            msg_queue.put(("update_totals", config.applied["epochs"], total_batches))
            msg_queue.put(("refresh",))

            # === Batchループ ===
            epoch_loss = 0.0
            num_batches_done = 0
            loader_iter = iter(loader)
            batch_idx = 0

            while True:
                # Batch開始: batch-level pending を適用
                batch_applied = config.apply_batch_level()

                if "input_size" in batch_applied:
                    new_sz = config.applied["input_size"]
                    dataset.input_size = (new_sz, new_sz)
                    msg_queue.put(("log", current_epoch, batch_idx,
                                   f"input_size: →{new_sz}"))
                    if config.applied["num_workers"] > 0:
                        loader = DataLoader(
                            dataset, batch_size=config.applied["batch_size"],
                            shuffle=True, collate_fn=collate_batch,
                            num_workers=config.applied["num_workers"])
                        total_batches = len(loader)
                        msg_queue.put(("update_totals",
                                       config.applied["epochs"], total_batches))
                        loader_iter = iter(loader)

                if "freeze_mode" in batch_applied:
                    _apply_freeze_mode(model, config.applied["freeze_mode"])
                    msg_queue.put(("log", current_epoch, batch_idx,
                                   f"freeze: →{config.applied['freeze_mode']}"
                                   " (optimizer: 次epoch)"))
                    pending_optimizer_rebuild = True

                if batch_applied:
                    msg_queue.put(("refresh",))

                try:
                    images, targets = next(loader_iter)
                except StopIteration:
                    break

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches_done += 1
                avg_loss_so_far = epoch_loss / num_batches_done

                msg_queue.put(("progress", current_epoch, batch_idx,
                               avg_loss_so_far, best_loss))

                # 停止チェック (バッチ境界)
                is_stop, is_cancel = stop_check_fn()
                if is_cancel:
                    cancelled = True
                    break
                if is_stop:
                    stopped_early = True
                    break

                batch_idx += 1

            if num_batches_done > 0:
                avg_loss = epoch_loss / num_batches_done
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = copy.deepcopy(model.state_dict())
                msg_queue.put(("epoch_done", current_epoch,
                               config.applied["epochs"], avg_loss, best_loss))

            if stopped_early or cancelled:
                break

            scheduler.step()
            current_epoch += 1

        # === 結果処理 ===
        if cancelled:
            msg_queue.put(("done", "cancelled", None, None, False))
        elif best_state is not None:
            torch.save(best_state, output_model_path)
            msg_queue.put(("done", "saved", output_model_path,
                           best_loss, stopped_early))
        else:
            msg_queue.put(("done", "no_data", None, None, False))

    except Exception as e:
        msg_queue.put(("error", str(e), traceback.format_exc()))


def execute_finetune(base_model_path, annotations_path,
                     image_dir, output_model_path,
                     epochs=15, lr=0.0005, batch_size=2):
    """
    既存モデルをベースにファインチューニング。
    学習ループは別スレッドで実行し、GUIは応答可能な状態を維持する。
    TrainingControlGUI から設定変更が可能。
    """
    print(f"\n{'='*50}")
    print(f" ファインチューニング開始")
    print(f" ベースモデル: {base_model_path}")
    print(f" アノテーション: {annotations_path}")
    print(f" エポック数: {epochs}, 学習率: {lr}")
    print(f"{'='*50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- TrainConfig 初期化 ---
    config = TrainConfig(
        input_size=INPUT_SIZE[0],
        num_workers=0,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        freeze_mode="all",
    )

    dataset = FineTuneDataset(annotations_path, image_dir,
                              input_size=(config.applied["input_size"],) * 2,
                              augment=True)
    print(f"学習サンプル数: {len(dataset)}")
    if len(dataset) == 0:
        print("学習データが0件です。アノテーションを確認してください。")
        return

    # 初期バッチ数推定 (GUI表示用)
    initial_batches = max(
        (len(dataset) + config.applied["batch_size"] - 1)
        // config.applied["batch_size"], 1)

    model = build_detection_model()
    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.train()

    gui = FinetuneProgressGUI(config.applied["epochs"], initial_batches)
    ctrl = TrainingControlGUI(gui.root, config)
    print("学習設定コントロールウィンドウで実行中に設定変更が可能です")

    msg_queue = _queue.Queue()

    def stop_check():
        return gui.stop_requested, gui.cancel_requested

    # --- 学習スレッド起動 ---
    train_thread = threading.Thread(
        target=_train_worker,
        args=(model, dataset, device, config, msg_queue,
              stop_check, output_model_path),
        daemon=True,
    )
    train_thread.start()

    # --- キューポーリング (メインスレッド: 80ms間隔) ---
    def poll():
        try:
            for _ in range(50):  # 1回のpollで最大50メッセージ処理
                msg = msg_queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, epoch, batch, avg_loss, best_loss = msg
                    gui.update(epoch, batch, avg_loss, best_loss)
                elif kind == "log":
                    _, epoch, batch, text = msg
                    ctrl.add_log(epoch, batch, text)
                elif kind == "refresh":
                    ctrl.refresh()
                elif kind == "update_totals":
                    _, ep, bt = msg
                    gui.update_totals(ep, bt)
                elif kind == "epoch_done":
                    _, epoch, total_ep, avg_loss, best_loss = msg
                    print(f"  Epoch {epoch+1:3d}/{total_ep} | "
                          f"Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
                elif kind == "done":
                    _, result, out_path, best_loss, early = msg
                    try:
                        ctrl.win.destroy()
                    except Exception:
                        pass
                    if result == "cancelled":
                        print("\n学習が中止されました。モデルは保存されていません。")
                        gui.finish_cancelled()
                    elif result == "saved":
                        if early:
                            print(f"\n早期停止! ベストモデルを保存しました。")
                        else:
                            print(f"\nファインチューニング完了!")
                        print(f"保存先: {out_path}")
                        gui.finish(out_path, best_loss, early_stopped=early)
                    else:  # no_data
                        print("学習が実行されませんでした。")
                        gui.root.destroy()
                    return  # ポーリング終了
                elif kind == "error":
                    _, err_msg, tb = msg
                    print(f"\n学習エラー: {err_msg}\n{tb}")
                    try:
                        ctrl.win.destroy()
                    except Exception:
                        pass
                    gui.finish_cancelled()
                    return  # ポーリング終了
        except _queue.Empty:
            pass
        gui.root.after(80, poll)

    gui.root.after(80, poll)
    gui.root.mainloop()


# ==========================================
# メイン処理
# ==========================================
def main():
    multiprocessing.freeze_support()

    root = tk.Tk()
    root.withdraw()

    mode = None

    def select_mode(m):
        nonlocal mode
        mode = m
        dlg.destroy()

    dlg = tk.Toplevel(root)
    dlg.title("SEM解析 - ファインチューニングツール")
    dlg.geometry("480x360")
    dlg.configure(bg="#2b2b2b")
    dlg.resizable(False, False)

    tk.Label(dlg, text="モードを選択してください", bg="#2b2b2b", fg="white",
             font=("Arial", 14, "bold")).pack(pady=15)

    tk.Button(dlg, text="1. 検出 + レビュー",
              bg="#0066AA", fg="white", font=("Arial", 12), width=30,
              command=lambda: select_mode("detect_review")).pack(pady=5)
    tk.Label(dlg, text="画像を解析し、検出結果をGUIでvoid/crack/otherに分類",
             bg="#2b2b2b", fg="#AAA", font=("Arial", 9)).pack()

    tk.Button(dlg, text="2. レビュー再開",
              bg="#CC8800", fg="white", font=("Arial", 12), width=30,
              command=lambda: select_mode("resume_review")).pack(pady=5)
    tk.Label(dlg, text="中断保存したannotations.jsonからレビューを再開",
             bg="#2b2b2b", fg="#AAA", font=("Arial", 9)).pack()

    tk.Button(dlg, text="3. ファインチューニング",
              bg="#228B22", fg="white", font=("Arial", 12), width=30,
              command=lambda: select_mode("finetune")).pack(pady=5)
    tk.Label(dlg, text="保存済みアノテーションを使ってモデルを追加学習",
             bg="#2b2b2b", fg="#AAA", font=("Arial", 9)).pack()

    dlg.protocol("WM_DELETE_WINDOW", lambda: select_mode(None))
    dlg.grab_set()
    root.wait_window(dlg)

    if mode is None:
        root.destroy()
        return

    # ==========================================
    # モード1: 検出 + レビュー
    # ==========================================
    if mode == "detect_review":
        print("\n>> [モード1] 検出 + レビュー")
        print(">> モデルを選択...")
        model_path = filedialog.askopenfilename(
            title="モデル(.pth)を選択",
            filetypes=[("Model", "*.pth")],
            initialdir=os.getcwd()
        )
        if not model_path:
            root.destroy()
            return

        print(">> 画像フォルダを選択...")
        folder_path_str = filedialog.askdirectory(title="画像フォルダを選択")
        root.destroy()
        if not folder_path_str:
            return

        folder_path = Path(folder_path_str)
        output_dir = folder_path / "_outputs_finetune"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                 and not f.startswith("output_")]
        if not files:
            print("画像がありません。")
            return

        print(f"\n------------------------------------------------")
        print(f" ハイブリッド並列処理を開始します")
        print(f" 総枚数: {len(files)} 枚")
        print(f" RAW_SCORE_THRESH: {RAW_SCORE_THRESH} (全保持)")
        print(f" SCORE_THRESH: {SCORE_THRESH} (初期表示)")
        print(f" Device 1: NVIDIA Quadro P620 (CUDA)")
        print(f" Device 2: Intel Xeon CPU (OpenVINO)")
        print(f"------------------------------------------------")

        # 検出実行 (RAW_SCORE_THRESH以上を全保持)
        all_data = execute_detection(model_path, folder_path, files)

        # 検出結果の出力画像保存
        det_count = sum(len(e["detections"]) for e in all_data.values())
        det_above_thresh = sum(
            1 for e in all_data.values()
            for d in e["detections"]
            if d["raw"]["score"] is not None and d["raw"]["score"] >= SCORE_THRESH
        )
        print(f"\n合計検出数: {det_count} (うちスコア>={SCORE_THRESH}: {det_above_thresh})")

        for filename, entry in all_data.items():
            dets = entry["detections"]
            visible = [
                d for d in dets
                if d["raw"]["score"] is not None
                and d["raw"]["score"] >= SCORE_THRESH
            ]
            if not visible:
                continue
            try:
                img = Image.open(folder_path / filename).convert("RGB")
                img = ImageOps.exif_transpose(img)
                draw = ImageDraw.Draw(img)
                for det in visible:
                    b = det["box"]
                    color = "red" if det["raw"]["label"] == "void" else "cyan"
                    draw.rectangle(b, outline=color, width=3)
                    draw.text((b[0], b[1] - 12),
                              f"{det['raw']['label']} {det['raw']['score']:.2f}", fill=color)
                img.save(output_dir / f"output_{filename}", quality=80)
            except Exception:
                pass

        # CSV保存
        with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "num_detections", "labels", "scores", "boxes"])
            for filename in sorted(all_data.keys()):
                dets = all_data[filename]["detections"]
                above = [d for d in dets if d["raw"]["score"] is not None and d["raw"]["score"] >= SCORE_THRESH]
                writer.writerow([
                    filename,
                    len(above),
                    ",".join(d["raw"]["label"] for d in above),
                    ",".join(f"{d['raw']['score']:.2f}" for d in above),
                    ";".join(str(d["box"]) for d in above),
                ])

        # レビューGUI起動
        annotations_path = output_dir / "annotations.json"
        print(f"\nレビューGUIを起動します...")
        print(f"  - BBoxをクリックして選択 → V/C/O でラベル変更")
        print(f"  - Space/Enter で画像をレビュー済みにして次へ")
        print(f"  - Ctrl+A で全BBoxをother")
        print(f"  - B でBBox作成モード (ドラッグで矩形追加)")
        print(f"  - Ctrl+Z で元に戻す")
        print(f"  - H でヘルプ表示")

        ReviewGUI(folder_path, all_data, annotations_path)

    # ==========================================
    # モード2: レビュー再開
    # ==========================================
    elif mode == "resume_review":
        print("\n>> [モード2] レビュー再開")
        print(">> annotations.jsonを選択...")
        annotations_path = filedialog.askopenfilename(
            title="中断保存したannotations.jsonを選択",
            filetypes=[("JSON", "*.json")]
        )
        if not annotations_path:
            root.destroy()
            return

        print(">> 画像フォルダを選択...")
        folder_path_str = filedialog.askdirectory(title="画像フォルダを選択")
        root.destroy()
        if not folder_path_str:
            return

        # JSONから検出データを復元 (旧形式も自動変換)
        all_data = load_annotations(annotations_path)

        folder_path = Path(folder_path_str)

        # 画像フォルダとの整合チェック: 欠損画像を除外
        missing = [fn for fn in all_data if not (folder_path / fn).exists()]
        if missing:
            print(f"\n[警告] {len(missing)}件の画像がフォルダに見つかりません:")
            for fn in missing[:10]:
                print(f"  - {fn}")
            if len(missing) > 10:
                print(f"  ... 他 {len(missing) - 10}件")
            # 欠損画像を除外
            for fn in missing:
                del all_data[fn]
            print(f"  → 欠損画像はレビュー対象から除外しました")

        if not all_data:
            print("有効な画像データが0件です。フォルダとJSONの対応を確認してください。")
            root.destroy()
            return

        print(f"読み込み完了: {len(all_data)} 画像分のアノテーション")

        count_total = sum(len(e["detections"]) for e in all_data.values())
        count_reviewed = sum(1 for e in all_data.values() if e["review_state"]["reviewed"])
        count_other = sum(1 for e in all_data.values() for d in e["detections"]
                          if d["review"]["label"] == "other")
        print(f"  検出総数: {count_total}, レビュー済画像: {count_reviewed}, other: {count_other}")

        ReviewGUI(folder_path, all_data, annotations_path)

    # ==========================================
    # モード3: ファインチューニング
    # ==========================================
    elif mode == "finetune":
        print("\n>> [モード3] ファインチューニング")
        print(">> ベースモデルを選択...")
        base_model_path = filedialog.askopenfilename(
            title="ベースモデル(.pth)を選択",
            filetypes=[("Model", "*.pth")],
            initialdir=os.getcwd()
        )
        if not base_model_path:
            root.destroy()
            return

        print(">> アノテーションJSONを選択...")
        annotations_path = filedialog.askopenfilename(
            title="アノテーション(annotations.json)を選択",
            filetypes=[("JSON", "*.json")]
        )
        if not annotations_path:
            root.destroy()
            return

        print(">> 画像フォルダを選択 (アノテーションに対応する画像があるフォルダ)...")
        image_dir = filedialog.askdirectory(title="画像フォルダを選択")
        root.destroy()
        if not image_dir:
            return

        base_name = Path(base_model_path).stem
        output_model_path = Path(base_model_path).parent / f"{base_name}_finetuned.pth"

        try:
            epochs_input = input(f"エポック数を入力 (デフォルト: 15): ").strip()
            epochs = int(epochs_input) if epochs_input else 15
        except ValueError:
            epochs = 15

        print(f"\n学習中、設定コントロールウィンドウで input_size / freeze / lr 等を変更できます。")

        execute_finetune(
            base_model_path=base_model_path,
            annotations_path=annotations_path,
            image_dir=image_dir,
            output_model_path=str(output_model_path),
            epochs=epochs,
            lr=0.0005,
            batch_size=2,
        )

        print(f"\n完了! ファインチューニング済みモデル:")
        print(f"  {output_model_path}")
        print(f"\nこのモデルを SEM_cudaopenvino.py または本ツールのモード1で使用できます。")


if __name__ == "__main__":
    main()
