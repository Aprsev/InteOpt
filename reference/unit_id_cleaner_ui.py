from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.collections import PathCollection

matplotlib.use("TkAgg")


@dataclass
class CsvStats:
    file: Path
    before_rows: int
    after_rows: int


class UnitIdCleanerUI:
    def _on_list_remain_select(self, event=None) -> None:
        # Placeholder for listbox selection event. No action needed for now.
        pass

    def _on_list_selected_select(self, event=None) -> None:
        # Placeholder for selected listbox selection event. No action needed for now.
        pass
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Unit ID Cleaner")
        self.root.geometry("1200x760")

        self.a_csv_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.include_subdirs = tk.BooleanVar(value=False)

        self.df_a: pd.DataFrame | None = None
        self.id_column: str | None = None
        self.selected_ids: set[int] = set()

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax_left = self.fig.add_subplot(121)
        self.ax_right = self.fig.add_subplot(122, sharex=self.ax_left, sharey=self.ax_left)
        self.canvas: FigureCanvasTkAgg | None = None
        self.bar_to_id: dict[int, int] = {}
        self.spatial_mode = False
        self.spatial_df: pd.DataFrame | None = None
        self.heatmap_base = None
        self.latest_heatmap_left = None
        self.latest_heatmap_right = None
        self.pixel_map = {}
        self.id_color_map = {}
        self.elongation_map = {}
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="A.csv:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.a_csv_path, width=90).grid(row=0, column=1, padx=6, sticky="we")
        ttk.Button(top, text="Browse", command=self._browse_a_csv).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Load", command=self._load_a_csv).grid(row=0, column=3, padx=4)

        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.output_dir, width=90).grid(row=1, column=1, padx=6, pady=(8, 0), sticky="we")
        ttk.Button(top, text="Browse", command=self._browse_output_dir).grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Checkbutton(top, text="Include subfolders", variable=self.include_subdirs).grid(
            row=1, column=3, padx=4, pady=(8, 0), sticky="w"
        )

        top.grid_columnconfigure(1, weight=1)

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(content, padding=6)
        right = ttk.Frame(content, padding=6)
        content.add(left, weight=1)
        content.add(right, weight=3)

        ttk.Label(left, text="Remaining/Unselected (- Select)").pack(anchor="w")
        self.id_list = tk.Listbox(left, selectmode=tk.EXTENDED, exportselection=False, height=10)
        self.id_list.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.id_list.bind("<<ListboxSelect>>", self._on_list_remain_select)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_frame, text="Add to Delete \u2193", command=self._move_to_selected).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="\u2191 Retrieve to Remain", command=self._move_to_remain).pack(side=tk.RIGHT)
        
        ttk.Label(left, text="Selected (for auto Delete)").pack(anchor="w")
        self.id_list_selected = tk.Listbox(left, selectmode=tk.EXTENDED, exportselection=False, height=10)
        self.id_list_selected.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.id_list_selected.bind("<<ListboxSelect>>", self._on_list_selected_select)
        
        auto_frame = ttk.Labelframe(left, text="Auto-Select by Shape", padding=6)
        auto_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(auto_frame, text="Elongation Threshold").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.thresh_var = tk.DoubleVar(value=2.5)
        ttk.Scale(auto_frame, from_=1.0, to=10.0, variable=self.thresh_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(auto_frame, textvariable=self.thresh_var).grid(row=0, column=2, sticky="e", padx=2)
        ttk.Button(auto_frame, text="Auto Select", command=self._auto_select).grid(row=1, column=0, columnspan=3, pady=(4, 0))
        auto_frame.columnconfigure(1, weight=1)

        bottom_frame = ttk.Frame(left)
        bottom_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bottom_frame, text="Clear selection", command=self._clear_selection).pack(side=tk.LEFT)
        ttk.Button(bottom_frame, text="Delete selected IDs", command=self._delete_selected_ids).pack(side=tk.RIGHT)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        
        self.toolbar_frame = ttk.Frame(right)
        self.toolbar_frame.pack(fill=tk.X, side=tk.TOP, pady=2)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick_bar)
        self.canvas.mpl_connect("button_press_event", self._on_click_spatial)

        self.log_text = tk.Text(right, height=9, state=tk.DISABLED)
        self.log_text.pack(fill=tk.X, pady=(8, 0))

        self._log("Choose A.csv and click Load.")

    def _browse_a_csv(self) -> None:
        file_path = filedialog.askopenfilename(title="Choose A.csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.a_csv_path.set(file_path)
            if not self.output_dir.get().strip():
                self.output_dir.set(self._default_output_dir(Path(file_path)).as_posix())

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_dir.set(folder)

    def _default_output_dir(self, a_csv: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return a_csv.parent / f"cleaned"

    def _detect_id_column(self, df: pd.DataFrame) -> str:
        candidates = ["unit_id", "id", "Unit_ID", "ID"]
        for col in candidates:
            if col in df.columns:
                return col
        raise ValueError(
            "No id column found. Expected one of: unit_id, id, Unit_ID, ID. "
            f"Columns in file: {list(df.columns)}"
        )

    def _detect_id_column_from_header(self, columns: Iterable[str]) -> str:
        candidates = ["unit_id", "id", "Unit_ID", "ID"]
        columns_set = set(columns)
        for col in candidates:
            if col in columns_set:
                return col
        raise ValueError(
            "No id column found. Expected one of: unit_id, id, Unit_ID, ID. "
            f"Columns in file: {list(columns)}"
        )

    def _detect_first_existing_column(self, columns: Iterable[str], candidates: list[str]) -> str | None:
        columns_set = set(columns)
        for col in candidates:
            if col in columns_set:
                return col
        return None

    def _prepare_spatial_data(self) -> None:
        self.spatial_mode = False
        self.spatial_df = None
        self.heatmap_base = None
        self.pixel_map = {}
        self.elongation_map = {}
        if hasattr(self, "base_im_left"):
            del self.base_im_left
        if hasattr(self, "base_im_right"):
            del self.base_im_right

        if self.df_a is None or self.id_column is None:
            return

        cols = list(self.df_a.columns)
        x_col = self._detect_first_existing_column(cols, ["x", "X", "col", "column", "pix_x", "xpix", "width"])
        y_col = self._detect_first_existing_column(cols, ["y", "Y", "row", "height", "pix_y", "ypix"])
        val_col = self._detect_first_existing_column(cols, ["value", "val", "a", "A", "weight", "intensity", "signal"])

        if x_col is None or y_col is None:
            return

        spatial = pd.DataFrame(
            {
                "unit_id": pd.to_numeric(self.df_a[self.id_column], errors="coerce"),
                "x": pd.to_numeric(self.df_a[x_col], errors="coerce"),
                "y": pd.to_numeric(self.df_a[y_col], errors="coerce"),
            }
        )

        if val_col is not None:
            spatial["value"] = pd.to_numeric(self.df_a[val_col], errors="coerce").fillna(0.0)
        else:
            spatial["value"] = 1.0

        spatial = spatial.dropna(subset=["unit_id", "x", "y"])
        if spatial.empty:
            return

        spatial["x"] = spatial["x"].round().astype(int)
        spatial["y"] = spatial["y"].round().astype(int)
        spatial["unit_id"] = spatial["unit_id"].astype(int)

        self.xmin, self.xmax = spatial["x"].min(), spatial["x"].max()
        self.ymin, self.ymax = spatial["y"].min(), spatial["y"].max()
        
        w = self.xmax - self.xmin + 1
        h = self.ymax - self.ymin + 1
        self.heatmap_base = np.zeros((h, w), dtype=float)
        
        coords_vals = spatial[["x", "y", "unit_id", "value"]].values
        
        for row in coords_vals:
            x, y, uid, val = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            px, py = x - self.xmin, y - self.ymin
            
            if val > self.heatmap_base[py, px]:
                self.heatmap_base[py, px] = val
                
            key = (x, y)
            if key not in self.pixel_map:
                self.pixel_map[key] = []
            self.pixel_map[key].append((uid, val))

        for key, uids_vals in self.pixel_map.items():
            self.pixel_map[key] = sorted(uids_vals, key=lambda x: x[1], reverse=True)

        for uid, group in spatial.groupby("unit_id"):
            val = group["value"].values
            sum_val = val.sum()
            if sum_val <= 0:
                self.elongation_map[uid] = 1.0
                continue
                
            x = group["x"].values
            y = group["y"].values
            
            mx = (x * val).sum() / sum_val
            my = (y * val).sum() / sum_val
            
            dx = x - mx
            dy = y - my
            
            cxx = (dx * dx * val).sum() / sum_val
            cyy = (dy * dy * val).sum() / sum_val
            cxy = (dx * dy * val).sum() / sum_val
            
            trace = cxx + cyy
            det = cxx * cyy - cxy * cxy
            
            # semi-major and semi-minor axis squared
            diff = np.sqrt(trace**2 - 4 * det) if trace**2 >= 4 * det else 0
            l1 = (trace + diff) / 2.0
            l2 = (trace - diff) / 2.0
            
            if l2 > 0:
                elongation = np.sqrt(l1 / l2)
            else:
                elongation = 100.0 if l1 > 0 else 1.0
                
            self.elongation_map[uid] = elongation

        self.spatial_df = spatial
        self.spatial_mode = True

    def _read_a_csv_optimized(self, a_csv: Path) -> tuple[pd.DataFrame, str]:
        header = pd.read_csv(a_csv, nrows=0)
        cols = list(header.columns)
        id_col = self._detect_id_column_from_header(cols)
        val_col = self._detect_first_existing_column(cols, ["value", "val", "a", "A", "weight", "intensity", "signal"])

        # If no value column is detected, fallback to full read.
        if not val_col:
            return pd.read_csv(a_csv), id_col

        chunk_size = 250_000
        df_list = []
        first_id = None
        
        for chunk in pd.read_csv(a_csv, chunksize=chunk_size):
            if first_id is None and not chunk.empty:
                # Identify the ID for the very first row
                first_id = chunk[id_col].iloc[0]
            
            # Keep rows that belong to the first unit_id OR contain a non-zero value.
            val_series = pd.to_numeric(chunk[val_col], errors="coerce").fillna(0)
            
            mask = (chunk[id_col] == first_id) | (val_series != 0)
            if mask.any():
                df_list.append(chunk[mask])
                
        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
        else:
            final_df = pd.DataFrame(columns=cols)
            
        return final_df, id_col

    def _load_a_csv(self) -> None:
        path_str = self.a_csv_path.get().strip()
        if not path_str:
            messagebox.showerror("Missing input", "Please choose A.csv first.")
            return

        a_csv = Path(path_str)
        if not a_csv.exists():
            messagebox.showerror("File not found", f"Cannot find file:\n{a_csv}")
            return

        try:
            t0 = perf_counter()
            df, id_col = self._read_a_csv_optimized(a_csv)
            elapsed_ms = (perf_counter() - t0) * 1000
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.df_a = df
        self.id_column = id_col
        self.selected_ids.clear()
        self._prepare_spatial_data()

        if not self.output_dir.get().strip():
            self.output_dir.set(self._default_output_dir(a_csv).as_posix())

        self._refresh_id_lists()
        self._draw_plot()
        self._log(f"Loaded A.csv: {a_csv}")
        self._log(f"Detected id column: {self.id_column}")
        if self.spatial_mode:
            self._log("Visualization mode: spatial map")
        else:
            self._log("Visualization mode: bar chart (no spatial x/y columns detected)")
        self._log("Optimized load: kept first unit_id and non-zero values for others.")
        self._log(f"Load time: {elapsed_ms:.1f} ms")
        self._log(f"Rows: {len(df)} | Unique IDs: {df[self.id_column].nunique(dropna=True)}")

    def _auto_select(self) -> None:
        if self.df_a is None or self.id_column is None:
            return
        threshold = self.thresh_var.get()
        for uid in self._unique_ids(self.df_a, self.id_column):
            if self.elongation_map.get(uid, 1.0) > threshold:
                self.selected_ids.add(uid)
        
        self._refresh_id_lists()
        self._draw_plot()

    def _move_to_selected(self) -> None:
        picked = [int(self.id_list.get(i)) for i in self.id_list.curselection()]
        for u in picked:
            self.selected_ids.add(u)
        self._refresh_id_lists()
        self._draw_plot()

    def _move_to_remain(self) -> None:
        picked = [int(self.id_list_selected.get(i)) for i in self.id_list_selected.curselection()]
        for u in picked:
            self.selected_ids.discard(u)
        self._refresh_id_lists()
        self._draw_plot()

    def _refresh_id_lists(self) -> None:
        self.id_list.delete(0, tk.END)
        self.id_list_selected.delete(0, tk.END)
        
        all_ids = self._unique_ids(self.df_a, self.id_column)
        
        # Use green for all units, white text
        if not self.id_color_map and all_ids:
            for uid in all_ids:
                self.id_color_map[uid] = ("#00ff00", "white")
                
        idx_rem = 0
        idx_sel = 0
        for unit_id in all_ids:
            if unit_id in self.selected_ids:
                self.id_list_selected.insert(tk.END, str(unit_id))
                bg, fg = self.id_color_map.get(unit_id, ("#00ff00", "white"))
                self.id_list_selected.itemconfig(idx_sel, {"bg": bg, "fg": fg})
                idx_sel += 1
            else:
                self.id_list.insert(tk.END, str(unit_id))
                bg, fg = self.id_color_map.get(unit_id, ("#00ff00", "white"))
                self.id_list.itemconfig(idx_rem, {"bg": bg, "fg": fg})
                idx_rem += 1

    def _unique_ids(self, df: pd.DataFrame | None, id_col: str | None) -> list[int]:
        if df is None or id_col is None:
            return []

        raw = pd.to_numeric(df[id_col], errors="coerce").dropna().astype(int)
        return sorted(raw.unique().tolist())

    def _draw_plot(self) -> None:
        if self.spatial_mode:
            self._draw_spatial_plot()
            return

        self.ax.clear()
        self.bar_to_id.clear()

        if self.df_a is None or self.id_column is None:
            self.ax.set_title("No data loaded")
            self.canvas.draw_idle()
            return

        series = pd.to_numeric(self.df_a[self.id_column], errors="coerce").dropna().astype(int)
        if series.empty:
            self.ax.set_title("No valid unit_id values")
            self.canvas.draw_idle()
            return

        counts = series.value_counts().sort_index()
        x = np.arange(len(counts))
        colors = ["tomato" if int(unit_id) in self.selected_ids else "steelblue" for unit_id in counts.index]

        bars = self.ax.bar(x, counts.values, color=colors, picker=True)
        for idx, unit_id in enumerate(counts.index):
            self.bar_to_id[idx] = int(unit_id)

        tick_step = max(1, len(counts) // 20)
        shown = x[::tick_step]
        labels = [str(int(counts.index[i])) for i in shown]
        self.ax.set_xticks(shown)
        self.ax.set_xticklabels(labels, rotation=45, ha="right")

        self.ax.set_xlabel("unit_id")
        self.ax.set_ylabel("rows count in A.csv")
        self.ax.set_title("Click bars to select/deselect IDs (red = selected)")
        self.ax.grid(alpha=0.25, axis="y")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_spatial_plot(self) -> None:
        self.bar_to_id.clear()

        if self.heatmap_base is None:
            self.ax_left.clear()
            self.ax_right.clear()
            self.ax_left.set_title("No spatial data available")
            self.canvas.draw_idle()
            return

        extent = [self.xmin - 0.5, self.xmax + 0.5, self.ymax + 0.5, self.ymin - 0.5]
        h, w = self.heatmap_base.shape
        # Show left: only unselected units; right: only selected units
        all_ids = self._unique_ids(self.df_a, self.id_column)
        sel_set = set(self.selected_ids)
        rem_set = set(all_ids) - sel_set

        # Build two heatmaps: left (unselected), right (selected)
        heatmap_left = np.zeros_like(self.heatmap_base)
        heatmap_right = np.zeros_like(self.heatmap_base)
        for (x, y), uids_vals in self.pixel_map.items():
            px, py = x - self.xmin, y - self.ymin
            for uid, val in uids_vals:
                if uid in rem_set:
                    if val > heatmap_left[py, px]:
                        heatmap_left[py, px] = val
                if uid in sel_set:
                    if val > heatmap_right[py, px]:
                        heatmap_right[py, px] = val

        self.latest_heatmap_left = heatmap_left
        self.latest_heatmap_right = heatmap_right

        # Always redraw both axes to ensure right plot updates
        self.ax_left.clear()
        self.ax_right.clear()
        self.base_im_left = self.ax_left.imshow(heatmap_left, cmap='Greens', extent=extent, interpolation='nearest', origin='upper', vmin=0)
        self.base_im_right = self.ax_right.imshow(heatmap_right, cmap='Greens', extent=extent, interpolation='nearest', origin='upper', vmin=0)
        self.ax_left.set_facecolor("black")
        self.ax_right.set_facecolor("black")
        self.ax_left.set_aspect("equal")
        self.ax_left.set_xlabel("x")
        self.ax_left.set_ylabel("y")
        self.ax_left.set_title("Remaining (Unselected)")
        self.ax_right.set_aspect("equal")
        self.ax_right.set_xlabel("x")
        self.ax_right.set_ylabel("y")
        self.ax_right.set_title("Selected (To Delete)")
        self.fig.tight_layout()

        self.canvas.draw_idle()

    def _on_click_spatial(self, event) -> None:
        # Use ax_left for spatial plot interaction
        if not self.spatial_mode or event.inaxes != self.ax_left:
            return

        if hasattr(self, "toolbar") and self.toolbar.mode != "":
            # Do not select if the user is currently panning or zooming
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        px_x, px_y = int(round(x)), int(round(y))
        hits = self.pixel_map.get((px_x, px_y), [])
        if not hits:
            return

        # Only select the top-most unit at this pixel
        best_uid = max(hits, key=lambda item: item[1])[0]

        # Toggle selection (single selection)
        if best_uid in self.selected_ids:
            self.selected_ids.remove(best_uid)
        else:
            self.selected_ids.clear()
            self.selected_ids.add(best_uid)

        self._sync_listbox_with_selection()
        self._draw_plot()

    def _on_pick_bar(self, event) -> None:
        if self.df_a is None or self.id_column is None:
            return
        if self.spatial_mode:
            return

        if not hasattr(event, "ind") or len(event.ind) == 0:
            return

        bar_idx = int(event.ind[0])
        if bar_idx not in self.bar_to_id:
            return

        unit_id = self.bar_to_id[bar_idx]
        if unit_id in self.selected_ids:
            self.selected_ids.remove(unit_id)
        else:
            self.selected_ids.add(unit_id)

        self._sync_listbox_with_selection()
        self._draw_plot()

    def _on_listbox_select(self, _event=None) -> None:
        picked = set()
        for idx in self.id_list.curselection():
            text = self.id_list.get(idx)
            try:
                picked.add(int(text))
            except ValueError:
                continue
        self.selected_ids = picked
        self._draw_plot()

    def _sync_listbox_with_selection(self) -> None:
        self.id_list.selection_clear(0, tk.END)
        all_ids = [self.id_list.get(i) for i in range(self.id_list.size())]
        selected_as_text = {str(v) for v in self.selected_ids}
        for i, value in enumerate(all_ids):
            if value in selected_as_text:
                self.id_list.selection_set(i)

    def _clear_selection(self) -> None:
        self.selected_ids.clear()
        self.id_list.selection_clear(0, tk.END)
        self._draw_plot()

    def _delete_selected_ids(self) -> None:
        if self.df_a is None or self.id_column is None:
            messagebox.showerror("No data", "Please load A.csv first.")
            return

        ids = sorted(self.selected_ids)
        if not ids:
            messagebox.showwarning("No selection", "Please select one or more unit_id values.")
            return

        input_a = Path(self.a_csv_path.get().strip())
        output_dir_str = self.output_dir.get().strip()
        output_root = Path(output_dir_str) if output_dir_str else input_a.parent / f"cleaner_figures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_root.mkdir(parents=True, exist_ok=True)

        csv_paths = self._find_csv_files(input_a.parent, recursive=self.include_subdirs.get())
        if not csv_paths:
            messagebox.showerror("No CSV found", f"No CSV files found under:\n{input_a.parent}")
            return

        stats: list[CsvStats] = []
        failed: list[tuple[Path, str]] = []

        for src in csv_paths:
            try:
                df = pd.read_csv(src)
                if self.id_column in df.columns:
                    before = len(df)
                    keep = ~pd.to_numeric(df[self.id_column], errors="coerce").isin(ids)
                    filtered = df.loc[keep].copy()
                    filtered.to_csv(src, index=False)
                    stats.append(CsvStats(file=src.relative_to(input_a.parent), before_rows=before, after_rows=len(filtered)))
                else:
                    # Keep file unchanged if target id column does not exist.
                    df.to_csv(src, index=False)
                    stats.append(CsvStats(file=src.relative_to(input_a.parent), before_rows=len(df), after_rows=len(df)))
            except Exception as exc:
                failed.append((src, str(exc)))

        # Update loaded A.csv in memory so visualization reflects deletion immediately.
        keep_a = ~pd.to_numeric(self.df_a[self.id_column], errors="coerce").isin(ids)
        self.df_a = self.df_a.loc[keep_a].copy()
        self._prepare_spatial_data()
        self.selected_ids.clear()
        self._refresh_id_lists()
        self._draw_plot()

        saved_paths = self._save_current_figures(output_root)

        deleted_rows_total = sum(s.before_rows - s.after_rows for s in stats)
        self._log(f"Deleted IDs: {ids}")
        self._log(f"Figure output folder: {output_root}")
        self._log(f"Processed CSV files: {len(stats)}")
        self._log(f"Total removed rows: {deleted_rows_total}")
        for path in saved_paths:
            self._log(f"Saved figure: {path}")

        if failed:
            self._log(f"Failed files: {len(failed)}")
            for src, err in failed[:5]:
                self._log(f"  - {src.name}: {err}")
            messagebox.showwarning(
                "Completed with warnings",
                f"Finished with {len(failed)} failed files. Check log panel for details.",
            )
        else:
            messagebox.showinfo("Done", "CSV files were overwritten and figures were saved.")

    def _save_current_figures(self, output_root: Path) -> list[Path]:
        saved: list[Path] = []
        combined_path = output_root / "cleaner_view_combined.png"
        self.fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        saved.append(combined_path)

        if self.latest_heatmap_left is not None and self.latest_heatmap_right is not None:
            extent = [self.xmin - 0.5, self.xmax + 0.5, self.ymax + 0.5, self.ymin - 0.5]

            left_fig = Figure(figsize=(6, 5), dpi=150)
            left_ax = left_fig.add_subplot(111)
            left_ax.set_facecolor("black")
            left_ax.imshow(self.latest_heatmap_left, cmap="Greens", extent=extent, interpolation="nearest", origin="upper", vmin=0)
            left_ax.set_aspect("equal")
            left_ax.set_xlabel("x")
            left_ax.set_ylabel("y")
            left_ax.set_title("Remaining (Unselected)")
            left_fig.tight_layout()
            left_path = output_root / "cleaner_view_left.png"
            left_fig.savefig(left_path, dpi=150, bbox_inches="tight")
            saved.append(left_path)

            right_fig = Figure(figsize=(6, 5), dpi=150)
            right_ax = right_fig.add_subplot(111)
            right_ax.set_facecolor("black")
            right_ax.imshow(self.latest_heatmap_right, cmap="Greens", extent=extent, interpolation="nearest", origin="upper", vmin=0)
            right_ax.set_aspect("equal")
            right_ax.set_xlabel("x")
            right_ax.set_ylabel("y")
            right_ax.set_title("Selected (To Delete)")
            right_fig.tight_layout()
            right_path = output_root / "cleaner_view_right.png"
            right_fig.savefig(right_path, dpi=150, bbox_inches="tight")
            saved.append(right_path)

        return saved

    def _find_csv_files(self, folder: Path, recursive: bool) -> list[Path]:
        pattern = "**/*.csv" if recursive else "*.csv"
        return sorted(p for p in folder.glob(pattern) if p.is_file())

    def _log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = UnitIdCleanerUI(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
