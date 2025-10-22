#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PAIRS = [
    # (title, (vicon_col, non_vicon_col), y_label, filename)
    ("Linear Velocity X", ("vicon_lin_vx", "base_lin_vx"), "vx (m/s)", "lin_vel_x.png"),
    ("Linear Velocity Y", ("vicon_lin_vy", "base_lin_vy"), "vy (m/s)", "lin_vel_y.png"),
    ("Linear Velocity Z", ("vicon_lin_vz", "base_lin_vz"), "vz (m/s)", "lin_vel_z.png"),

    ("proj_gx",  ("proj_gx_vicon",  "proj_gx"),  "gravity_x",  "gravity_x.png"),
    ("proj_gy", ("proj_gy_vicon", "proj_gy"), "gravity_y", "gravity_y.png"),
    ("proj_gz",   ("proj_gz_vicon",   "proj_gz"),   "gravity_z",   "gravity_z.png"),

    ("Gyro X", ("gyro_x_vicon", "gyro_x"), "wx (rad/s)", "gyro_x.png"),
    ("Gyro Y", ("gyro_y_vicon", "gyro_y"), "wy (rad/s)", "gyro_y.png"),
    ("Gyro Z", ("gyro_z_vicon", "gyro_z"), "wz (rad/s)", "gyro_z.png"),
    ("height", ("body_height", "body_height"), "m", "height.png"),

]

def pick_time_axis(df: pd.DataFrame) -> np.ndarray:
    """优先使用 't' 列；若没有就用索引当作时间轴。"""
    if "t" in df.columns:
        t = df["t"].to_numpy()
        # 如果 t 有 NaN 或全常数，退化为 range
        if np.any(~np.isfinite(t)) or (np.max(t) - np.min(t) == 0):
            return np.arange(len(df), dtype=float)
        return t
    else:
        return np.arange(len(df), dtype=float)

def plot_pair(df: pd.DataFrame, t: np.ndarray, vicon_col: str, non_vicon_col: str,
              title: str, y_label: str, out_path: str) -> bool:
    """画一张对比图；返回是否成功绘图。"""
    ok = True
    missing = []
    if vicon_col not in df.columns:
        ok = False
        missing.append(vicon_col)
    if non_vicon_col not in df.columns:
        ok = False
        missing.append(non_vicon_col)

    if not ok:
        print(f"[Skip] {title}: 缺少列 {missing}")
        return False

    y1 = df[vicon_col].to_numpy()
    y2 = df[non_vicon_col].to_numpy()

    # 容错：长度不一致则裁剪到共同长度
    n = min(len(t), len(y1), len(y2))
    t, y1, y2 = t[:n], y1[:n], y2[:n]

    plt.figure()
    plt.plot(t, y1, label=vicon_col)
    plt.plot(t, y2, label=non_vicon_col)
    plt.title(title)
    plt.xlabel("time (s) or index")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] {title} -> {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Plot Vicon vs Non-Vicon signals from CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the saved CSV (e.g., b1_run_*.csv)")
    parser.add_argument("--out", type=str, default="figs_vicon_vs_base", help="Output directory for figures")
    parser.add_argument("--start", type=int, default=0, help="Slice start index")
    parser.add_argument("--end", type=int, default=None, help="Slice end index (exclusive)")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (>=1)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 读 CSV（自动解析 header 中的中文/英文都没问题）
    df = pd.read_csv(args.csv_path)

    # 切片 + 下采样（便于长日志）
    df = df.iloc[args.start:args.end:args.downsample].reset_index(drop=True)

    t = pick_time_axis(df)

    any_ok = False
    for title, (v_col, n_col), ylab, fname in PAIRS:
        out_path = os.path.join(args.out, fname)
        ok = plot_pair(df, t, v_col, n_col, title, ylab, out_path)
        any_ok = any_ok or ok

    if not any_ok:
        print("没有找到可用的列名，请确认你的 CSV header 与脚本中的列名一致。")
        print("当前 CSV 列名示例：", list(df.columns)[:40], "...")
        print("如果你的列名不同，请修改脚本中 PAIRS 的列名映射。")

if __name__ == "__main__":
    main()
