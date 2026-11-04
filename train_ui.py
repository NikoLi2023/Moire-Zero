# -*- coding: utf-8 -*-
"""MZNet 训练 UI：可视化设置训练参数，一键启动 quick_train.py，并把训练过程的
Print 输出实时呈现在界面上。

用法（在仓库根目录、使用训练环境 python 运行）：

    python train_ui.py

说明：
  - 必填：训练目录、任务权重名、总轮数 epochs、结果目录（默认 D:\DemoireModel）；
    任务权重名同时作为任务名与最终权重名；
  - 训练目录：自行选择/输入文件夹路径（"浏览…"弹出文件夹选择框），
    该目录须内含 train、test 子文件夹；启动时自动拆成 数据父目录 + 目录名，
    train_dataset = val_dataset = 目录名（训练集读 train、验证集读 test）；
  - 结果目录：可自行修改，默认 D:\DemoireModel；最终权重保存在
    <结果目录>/<训练集目录名>/ckpt/<任务权重名>.pth.tar；
  - 选填：学习率 lr（默认 0.0008）、批量 batch_size（默认 2）；
    基础配置固定使用 configs/GQMoire.yml，不在界面开放；
    初始权重不在界面配置，由基础配置 yml 的 training.resume 决定；
  - 训练结束后始终自动导出 fp16 ONNX 到 ckpt 文件夹的上一级（训练集目录），
    即 <结果目录>/<训练集目录名>/<任务权重名>.onnx，无需勾选；
  - ONNX 导出后自动调用项目目录下 ModelCrypto.dll 加密为
    "<任务权重名>.demoire.qgAi"（与 ONNX 同目录），
    并删除原 .onnx（加密失败也同样删除并记录原因，见 model_crypto.py）；
  - 训练在独立子进程中运行（与 UI 解耦，停止按钮可安全终止），stdout/stderr 实时流式显示；
  - 界面日志中的 tqdm 进度条按行刷新显示。
"""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 仓库根目录（quick_train.py 所在处）

# 日志固定目录：不存在时自动创建；文件名 DemoireTrain-YYYYMMDD.txt
LOG_DIR = r'C:\GQI Data\OptiWorks\UserData'

# 软件版本号（显示在状态栏右下角）
APP_VERSION = '1.0.0.1'

# ---------------- 深色主题配色（参照 GQ 系工具：深紫标题栏 + 深灰卡片 + 蓝/绿点缀） ----------------
C_BG       = '#1E1F24'   # 窗口主背景
C_CARD     = '#2A2B31'   # 分组卡片背景
C_CARD_BD  = '#3A3B42'   # 卡片边框
C_HDR      = '#2A2340'   # 标题栏（深紫）
C_HDR_SUB  = '#9C93B8'   # 标题栏副标题
C_ENTRY    = '#33343B'   # 输入框背景
C_TEXT     = '#E6E6E6'   # 主文字
C_TEXT_DIM = '#9AA0A6'   # 次要文字
C_ACCENT   = '#4A6CF7'   # 蓝（主按钮 / 强调）
C_GREEN    = '#3BA55C'   # 绿（成功）
C_RED      = '#ED4245'   # 红（错误 / 停止）
C_LOG_BG   = '#16171B'   # 日志区背景
C_REQ      = '#FF7A7A'   # 必填标记
C_BTN_BG   = '#404659'   # 次级按钮（浏览等）底色：比卡片亮一档
C_BTN_HOV  = '#525B73'   # 次级按钮悬停
FONT_UI    = 'Microsoft YaHei UI'


class DailyLogWriter(threading.Thread):
    """按天写日志的异步日志器：调用方只把文本放入队列（不阻塞），
    由独立后台线程攒批写入 <log_dir>/DemoireTrain-YYYYMMDD.txt。

    - 缓冲阈值 4KB 或 2 秒超时即落盘，行首自动加时间戳；
    - 跨天时自动切换到新日期的文件；
    - stop() 发送哨兵，线程冲刷剩余缓冲后退出。
    """

    FLUSH_SIZE = 4096
    FLUSH_TIMEOUT = 2.0

    def __init__(self, log_dir):
        super().__init__(daemon=True, name='DailyLogWriter')
        self.log_dir = log_dir
        self.q = queue.Queue()
        self._buf = []
        self._buf_size = 0
        self._pending = ''      # 未凑满一行的残尾
        self._fh = None
        self._day = None
        os.makedirs(log_dir, exist_ok=True)
        self.start()

    def write(self, text):
        """非阻塞：仅入队。即使写入端繁忙也不会拖慢调用线程。"""
        self.q.put(text)

    def stop(self, timeout=3.0):
        self.q.put(None)
        self.join(timeout=timeout)

    def _open_for_today(self):
        day = time.strftime('%Y-%m-%d')
        if day != self._day:
            if self._fh is not None:
                self._fh.close()
            path = os.path.join(self.log_dir, f'DemoireTrain-{time.strftime("%Y%m%d")}.txt')
            self._fh = open(path, 'a', encoding='utf-8')
            self._day = day

    def _flush(self):
        if not self._buf:
            return
        self._pending += ''.join(self._buf)
        self._buf.clear()
        self._buf_size = 0
        lines = self._pending.split('\n')
        self._pending = lines.pop()          # 末尾可能是不完整行，留待下次
        if not lines:
            return
        try:
            self._open_for_today()
        except OSError:
            return                            # 磁盘异常时丢弃该批，不影响主流程
        ts = time.strftime('%H:%M:%S')
        for line in lines:
            self._fh.write(f'{ts} {line}\n' if line.strip() else '\n')
        self._fh.flush()

    def run(self):
        stop = False
        while not stop:
            try:
                item = self.q.get(timeout=self.FLUSH_TIMEOUT)
            except queue.Empty:
                self._flush()                 # 超时：把已积累的刷出去
                continue
            if item is None:
                stop = True
            else:
                self._buf.append(item)
                self._buf_size += len(item)
            # 不阻塞地取空当前积压
            while True:
                try:
                    extra = self.q.get_nowait()
                except queue.Empty:
                    break
                if extra is None:
                    stop = True
                    continue
                self._buf.append(extra)
                self._buf_size += len(extra)
            if self._buf_size >= self.FLUSH_SIZE:
                self._flush()
        self._flush()
        if self._fh is not None:
            self._fh.close()


class TrainUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('GQDemoire 训练控制台')
        self.geometry('960x800+120+80')
        self.config(bg=C_BG)
        # 去掉系统标题栏：最小化/最大化/关闭由自定义标题栏提供（见 _build_header）
        self.overrideredirect(True)
        self._maximized = False
        self._normal_geo = None
        self._drag = (0, 0)

        self.proc = None            # 训练子进程
        self.log_queue = queue.Queue()
        self.start_time = None
        self._input_widgets = []    # 运行时禁用的输入控件
        self._logo_img = None       # 引用 Logo，防止被 GC 回收

        # 窗口图标与标题栏 Logo 使用同一张图
        try:
            self._logo_img = tk.PhotoImage(file=os.path.join(SCRIPT_DIR, 'ICO', 'Logo.png'))
            self.iconphoto(True, self._logo_img)
        except Exception:
            self._logo_img = None

        # 浏览对话框的起始位置：记忆上次选择，跨会话保存
        self._last_browse_dir = self._load_last_browse_dir()

        # 异步日志器：界面显示的所有内容按天落盘到固定日志目录（不阻塞 UI）
        self.log_writer = DailyLogWriter(LOG_DIR)
        self.protocol('WM_DELETE_WINDOW', self._on_close)   # 关窗口时冲刷日志
        self.bind('<Map>', self._on_mapped)                 # 最小化还原后恢复无边框样式

        # pack 顺序：状态栏最先布局（side='bottom' 先抢占底部条带），
        # 防止日志区(expand)把底部的运行状态挤到窗口外
        self._build_statusbar()
        self._build_header()
        self._build_params()
        self._build_buttons()
        self._build_log()

        self.append_log(f'===== 会话开始 {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n', 'sys')

        self.after(100, self._poll_log)

    # ---------------- 标题栏（自定义窗口标题栏：Logo + 标题 + 最小化/最大化/关闭） ----------------
    def _win_control(self, parent, kind, command):
        """窗口控制按钮：用 Canvas 绘制线性图标（最小化/最大化/关闭），悬停提亮、关闭悬停变红。"""
        W, H = 46, 42
        cv = tk.Canvas(parent, width=W, height=H, bg=C_HDR,
                       highlightthickness=0, cursor='hand2')

        def _draw(color='#CFC8E3'):
            cv.delete('all')
            if kind == 'min':
                cv.create_line(17, H - 14, 29, H - 14, fill=color, width=1.6)
            elif kind == 'close':
                cv.create_line(17, 14, 29, 26, fill=color, width=1.6)
                cv.create_line(29, 14, 17, 26, fill=color, width=1.6)
            else:  # max / restore
                if self._maximized:
                    cv.create_rectangle(19, 11, 30, 22, outline=color, width=1.6)
                    cv.create_rectangle(15, 15, 26, 26, outline=color, width=1.6)
                else:
                    cv.create_rectangle(17, 13, 29, 25, outline=color, width=1.6)

        hover_bg = C_RED if kind == 'close' else '#463D63'
        cv.bind('<Enter>', lambda _e: (cv.config(bg=hover_bg), _draw('#FFFFFF')))
        cv.bind('<Leave>', lambda _e: (cv.config(bg=C_HDR), _draw()))
        cv.bind('<Button-1>', lambda _e: command())
        cv.pack(side='right')
        _draw()   # 初始绘制，避免打开窗口时按钮图标空白（仅在悬停后才显示）
        if kind == 'max':
            self._max_icon_redraw = _draw   # 最大化/还原后刷新图标
        return cv

    def _minimize(self):
        # override-redirect 窗口无法直接 iconify：临时恢复系统边框最小化，
        # 还原时由 <Map> 事件把无边框样式补回来（见 _on_mapped）
        self.overrideredirect(False)
        self.iconify()

    def _on_mapped(self, _e):
        if not self._maximized:
            try:
                self.overrideredirect(True)
            except tk.TclError:
                pass

    def _toggle_max(self):
        if self._maximized:
            self.state('normal')
            if self._normal_geo:
                self.geometry(self._normal_geo)
            self._maximized = False
        else:
            self._normal_geo = self.geometry()
            try:
                self.state('zoomed')
            except tk.TclError:
                w = self.winfo_screenwidth()
                h = self.winfo_screenheight()
                self.geometry(f'{w}x{h}+0+0')
            self._maximized = True
        redraw = getattr(self, '_max_icon_redraw', None)
        if redraw:
            redraw()

    def _start_drag(self, e):
        self._drag = (e.x_root - self.winfo_x(), e.y_root - self.winfo_y())

    def _on_drag(self, e):
        if self._maximized:
            return
        dx, dy = self._drag
        self.geometry(f'+{e.x_root - dx}+{e.y_root - dy}')

    def _bind_drag(self, w):
        w.bind('<Button-1>', self._start_drag)
        w.bind('<B1-Motion>', self._on_drag)

    def _build_header(self):
        header = tk.Frame(self, bg=C_HDR)
        header.pack(fill='x', side='top')

        # Logo：缩放到标题栏高度（ICO/Logo.png）
        if self._logo_img is not None:
            try:
                factor = max(1, round(self._logo_img.height() / 38))
                img = self._logo_img.subsample(factor, factor)
                lbl_logo = tk.Label(header, image=img, bg=C_HDR)
                lbl_logo.pack(side='left', padx=(14, 8), pady=9)
                self._logo_img_small = img   # 防止被 GC 回收
            except Exception:
                pass

        title_box = tk.Frame(header, bg=C_HDR)
        title_box.pack(side='left', pady=7)
        lbl_title = tk.Label(title_box, text='GQDemoire 训练控制台', bg=C_HDR, fg='#FFFFFF',
                             font=(FONT_UI, 14, 'bold'))
        lbl_title.pack(anchor='w')
        tk.Label(title_box, text='Moiré Removal Training Console', bg=C_HDR, fg=C_HDR_SUB,
                 font=(FONT_UI, 9)).pack(anchor='w')

        # 窗口控制按钮：最小化 / 最大化(还原) / 关闭（替代系统标题栏按钮）
        self._win_control(header, 'close', self._on_close)
        self._win_control(header, 'max', self._toggle_max)
        self._win_control(header, 'min', self._minimize)

        # 标题栏可拖动窗口
        for w in (header, title_box, lbl_title):
            self._bind_drag(w)
        if self._logo_img is not None:
            try:
                self._bind_drag(lbl_logo)
            except Exception:
                pass

        # 右下角拖拽缩放手柄（无边框窗口失去系统调整边框，手动补一个）
        grip = tk.Frame(self, bg=C_CARD_BD, cursor='size_nw_se', width=16, height=16)
        grip.place(relx=1.0, rely=1.0, x=-2, y=-2, anchor='se')

        def _grip_start(e):
            grip._gxy = (e.x_root, e.y_root)

        def _grip_move(e):
            gx, gy = getattr(grip, '_gxy', (e.x_root, e.y_root))
            new_w = max(760, self.winfo_width() + (e.x_root - gx))
            new_h = max(600, self.winfo_height() + (e.y_root - gy))
            grip._gxy = (e.x_root, e.y_root)
            self.geometry(f'{new_w}x{new_h}')

        grip.bind('<Button-1>', _grip_start)
        grip.bind('<B1-Motion>', _grip_move)

    def _on_close(self):
        """关窗口：停止日志线程（冲刷剩余缓冲）后退出。"""
        try:
            self.log_writer.stop()
        except Exception:
            pass
        self.destroy()

    # ---------------- 界面布局 ----------------
    def _make_entry(self, parent, var, width=42):
        """深色输入框（运行禁用时保持配色）。"""
        ent = tk.Entry(parent, textvariable=var, width=width,
                       bg=C_ENTRY, fg=C_TEXT, insertbackground=C_TEXT,
                       relief='flat', highlightthickness=1,
                       highlightcolor=C_ACCENT, highlightbackground=C_CARD_BD,
                       disabledbackground=C_ENTRY, disabledforeground='#8A8A8A')
        self._input_widgets.append(ent)
        return ent

    def _make_button(self, parent, text, command, bg=C_CARD, fg=C_TEXT, hover=None):
        """扁平深色按钮；hover 时轻微提亮。"""
        b = tk.Button(parent, text=text, command=command,
                      bg=bg, fg=fg, relief='flat', cursor='hand2',
                      activebackground=hover or bg, activeforeground='#FFFFFF',
                      disabledforeground='#7A7A7A',
                      font=(FONT_UI, 9), padx=10, pady=3)
        b.bind('<Enter>', lambda _e: b.config(bg=hover or bg) if str(b['state']) != 'disabled' else None)
        b.bind('<Leave>', lambda _e: b.config(bg=bg) if str(b['state']) != 'disabled' else None)
        return b

    def _build_params(self):
        # ---------------- 必填项 ----------------
        req = tk.LabelFrame(self, text=' 必填项 ', bg=C_CARD, fg=C_ACCENT,
                            font=(FONT_UI, 10, 'bold'),
                            relief='solid', bd=1, labelanchor='nw')
        req.pack(fill='x', padx=10, pady=(10, 4))

        self.var_train = tk.StringVar()
        self.var_name = tk.StringVar()
        self.var_epochs = tk.StringVar()
        self.var_result = tk.StringVar(value=r'D:\DemoireModel')

        # 训练目录：自行选择/输入文件夹路径（内含 train、test 子文件夹）
        tk.Label(req, text='训练目录 *', bg=C_CARD, fg=C_REQ,
                 font=(FONT_UI, 9)).grid(row=0, column=0, sticky='e', padx=6, pady=4)
        ent_train = self._make_entry(req, self.var_train, width=52)
        ent_train.grid(row=0, column=1, sticky='w', padx=4, pady=4)
        self._make_button(req, '浏览…', self._browse_train_dir,
                          bg=C_BTN_BG, fg=C_TEXT, hover=C_BTN_HOV).grid(row=0, column=2, sticky='w', padx=4)

        for r, (label, var) in enumerate((('任务权重名 *', self.var_name),
                                          ('总轮数 epochs *', self.var_epochs)), start=1):
            tk.Label(req, text=label, bg=C_CARD, fg=C_REQ,
                     font=(FONT_UI, 9)).grid(row=r, column=0, sticky='e', padx=6, pady=4)
            ent = self._make_entry(req, var)
            ent.grid(row=r, column=1, sticky='w', padx=4, pady=4)

        # 结果目录：训练权重与 ONNX 输出位置（可设置，不再固定 results/）
        tk.Label(req, text='结果目录 *', bg=C_CARD, fg=C_REQ,
                 font=(FONT_UI, 9)).grid(row=3, column=0, sticky='e', padx=6, pady=4)
        ent_result = self._make_entry(req, self.var_result, width=52)
        ent_result.grid(row=3, column=1, sticky='w', padx=4, pady=4)
        self._make_button(req, '浏览…', self._browse_result_dir,
                          bg=C_BTN_BG, fg=C_TEXT, hover=C_BTN_HOV).grid(row=3, column=2, sticky='w', padx=4)

        # ---------------- 可选项 ----------------
        opt = tk.LabelFrame(self,
                            text=' 可选项',
                            bg=C_CARD, fg=C_TEXT_DIM, font=(FONT_UI, 10, 'bold'),
                            relief='solid', bd=1, labelanchor='nw')
        opt.pack(fill='x', padx=10, pady=4)

        self.var_lr = tk.StringVar(value='0.0008')
        self.var_batch = tk.StringVar(value='2')
        # 基础配置不在界面开放：固定使用 configs/GQMoire.yml
        self.var_config = tk.StringVar(value='GQMoire.yml')

        def row(parent, r, label, var, width=42):
            tk.Label(parent, text=label, bg=C_CARD, fg=C_TEXT_DIM,
                     font=(FONT_UI, 9)).grid(row=r, column=0, sticky='e', padx=6, pady=4)
            ent = self._make_entry(parent, var, width=width)
            ent.grid(row=r, column=1, sticky='w', padx=4, pady=4)
            return ent

        row(opt, 0, '学习率 lr', self.var_lr)
        row(opt, 1, '批量 batch_size', self.var_batch)

    # ---------------- 浏览位置记忆 ----------------
    _LAST_DIR_FILE = os.path.join(SCRIPT_DIR, '.train_ui_lastdir')

    def _load_last_browse_dir(self):
        try:
            with open(self._LAST_DIR_FILE, 'r', encoding='utf-8') as f:
                p = f.read().strip()
            if p and os.path.isdir(p):
                return p
        except OSError:
            pass
        return SCRIPT_DIR

    def _save_last_browse_dir(self, path):
        self._last_browse_dir = path
        try:
            with open(self._LAST_DIR_FILE, 'w', encoding='utf-8') as f:
                f.write(path)
        except OSError:
            pass

    def _browse_dir(self, title, var):
        """弹出文件夹选择框，起始位置为上次选择的文件夹；选定后记忆该位置。"""
        p = filedialog.askdirectory(initialdir=self._last_browse_dir, title=title)
        if p:
            var.set(p)
            self._save_last_browse_dir(p)

    def _browse_train_dir(self):
        self._browse_dir('选择训练目录（内含 train/test 文件夹）', self.var_train)

    def _browse_result_dir(self):
        self._browse_dir('选择结果目录（训练权重与 ONNX 输出位置）', self.var_result)

    def _open_result_dir(self):
        p = self.var_result.get().strip() or os.path.join(SCRIPT_DIR, 'results')
        if os.path.isdir(p):
            os.startfile(p)
        elif hasattr(self, 'status_var'):
            self.status_var.set(f'结果目录不存在：{p}')

    def _build_buttons(self):
        # 按钮行（在日志区之前 pack，保证窗口打开时始终可见）
        btns = tk.Frame(self, bg=C_BG)
        btns.pack(fill='x', padx=10, pady=4)
        self.btn_start = self._make_button(btns, '▶ 开始训练', self.start_train,
                                           bg=C_ACCENT, fg='#FFFFFF', hover='#5B7DFF')
        self.btn_start.pack(side='left', padx=2)
        self.btn_stop = self._make_button(btns, '■ 停止', self.stop_train,
                                          bg=C_RED, fg='#FFFFFF', hover='#F25C5F',
                                          )
        self.btn_stop.config(state='disabled')
        self.btn_stop.pack(side='left', padx=2)
        self._make_button(btns, '清空日志', self.clear_log,
                          bg=C_BTN_BG, fg=C_TEXT, hover=C_BTN_HOV).pack(side='left', padx=2)
        self._make_button(btns, '打开结果目录', self._open_result_dir,
                          bg=C_BTN_BG, fg=C_TEXT, hover=C_BTN_HOV).pack(side='left', padx=2)
        # 与底部状态栏同源的运行状态，放在按钮行保证任何窗口高度下都可见
        tk.Label(btns, textvariable=self.status_var, bg=C_BG, fg=C_ACCENT,
                 font=(FONT_UI, 9)).pack(side='left', padx=14)

    def _build_log(self):
        frame = tk.LabelFrame(self, text=' 训练输出 ', bg=C_CARD, fg=C_TEXT_DIM,
                              font=(FONT_UI, 10, 'bold'),
                              relief='solid', bd=1, labelanchor='nw')
        frame.pack(fill='both', expand=True, padx=10, pady=(4, 8))
        self.log_text = scrolledtext.ScrolledText(
            frame, wrap='none', state='normal',
            bg=C_LOG_BG, fg=C_TEXT, insertbackground=C_TEXT,
            font=('Consolas', 10), relief='flat')
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)
        self.log_text.tag_config('stderr', foreground=C_RED)
        self.log_text.tag_config('sys', foreground='#5B8CFF')

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value='就绪')
        bar = tk.Frame(self, bg=C_CARD)
        bar.pack(fill='x', side='bottom')
        tk.Frame(bar, bg=C_CARD_BD, height=1).pack(fill='x')
        tk.Label(bar, textvariable=self.status_var, bg=C_CARD, fg=C_TEXT_DIM,
                 font=(FONT_UI, 9)).pack(side='left', padx=8, pady=3)
        tk.Label(bar, text=f'Version: {APP_VERSION}', bg=C_CARD, fg='#6E7078',
                 font=(FONT_UI, 9)).pack(side='right', padx=28, pady=3)   # 右侧让出缩放手柄位置

    # ---------------- 日志 ----------------
    def append_log(self, text, tag=None):
        self.log_text.insert('end', text, tag)
        self.log_text.see('end')
        self.log_writer.write(text)   # 异步落盘：仅入队，不阻塞

    def clear_log(self):
        self.log_text.delete('1.0', 'end')

    def _poll_log(self):
        """UI 线程定时从队列取日志，避免跨线程直接操作控件。"""
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item is None:          # 子进程结束哨兵
                    self._on_training_end()
                    break
                text, is_err = item
                self.append_log(text, 'stderr' if is_err else None)
        except queue.Empty:
            pass
        # 运行中刷新状态栏计时
        if self.proc is not None and self.start_time:
            el = time.time() - self.start_time
            self.status_var.set(f'训练运行中… 已用 {int(el // 60)} 分 {int(el % 60)} 秒')
        self.after(100, self._poll_log)

    # ---------------- 训练控制 ----------------
    def _validate(self):
        train = self.var_train.get().strip()
        result = self.var_result.get().strip()
        task_name = self.var_name.get().strip()
        lr = self.var_lr.get().strip()
        epochs = self.var_epochs.get().strip()
        if not train:
            return None, '请选择/输入训练目录（内含 train、test 文件夹的文件夹路径）'
        if not os.path.isdir(train):
            return None, f'训练目录不存在：{train}'
        for sub in ('train', 'test'):
            if not os.path.isdir(os.path.join(train, sub)):
                return None, f'训练目录下缺少 {sub} 子文件夹：{train}'
        if not result:
            return None, '请选择/输入结果目录（训练权重与 ONNX 输出位置）'
        if not task_name:
            return None, '请填写任务权重名'
        lr_value = None
        if lr:
            try:
                lr_value = float(lr)
                if lr_value <= 0:
                    raise ValueError
            except ValueError:
                return None, f'学习率 lr 必须是正数，当前值：{lr!r}'
        try:
            ep_v = int(epochs)
            if ep_v <= 0:
                raise ValueError
        except ValueError:
            return None, f'epochs 必须是正整数，当前值：{epochs!r}'
        return {
            'train_dataset': train,
            'result_folder': result,
            'lr': lr_value,
            'epochs': epochs,
            'task_name': task_name,
        }, None

    def start_train(self):
        params, err = self._validate()
        if err:
            self.append_log(f'[UI] 参数错误：{err}\n', 'sys')
            return

        # 训练目录 -> 拆成 数据父目录 + 数据集目录名（train_dataset = val_dataset）
        train_dir = os.path.normpath(params['train_dataset'])
        data_parent, ds_name = os.path.split(train_dir)
        cmd = [sys.executable, '-u', 'quick_train.py',
               '--train_dataset', ds_name,
               '--val_dataset', ds_name,
               '--epochs', params['epochs'],
               '--name', params['task_name'],
               '--ckpt_name', params['task_name'],
               '--result_folder', params['result_folder']]
        if data_parent:
            cmd += ['--data_dir', data_parent]
        if params['lr'] is not None:
            cmd += ['--lr', str(params['lr'])]
        if self.var_config.get().strip():
            cmd += ['--config', self.var_config.get().strip()]
        if self.var_batch.get().strip():
            cmd += ['--batch_size', self.var_batch.get().strip()]
        # 不传 --resume：初始权重由基础配置 yml 的 training.resume 决定
        cmd.append('--export_onnx')   # 训练结束后默认始终自动导出 fp16 ONNX

        try:
            os.makedirs(params['result_folder'], exist_ok=True)
        except OSError as e:
            self.append_log(f'[UI] 结果目录创建失败：{e}\n', 'stderr')
            return

        # self.append_log('[UI] 启动训练：' + ' '.join(cmd) + '\n', 'sys')

        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'   # 训练脚本中文输出统一按 utf-8 读取，避免乱码
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=SCRIPT_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception as e:
            self.append_log(f'[UI] 启动失败：{e}\n', 'stderr')
            self.proc = None
            return

        self.start_time = time.time()
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        for w in self._input_widgets:
            w.config(state='disabled')
        self.status_var.set('训练启动中…')

        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self):
        """后台线程读子进程输出（os.read 立即返回已有数据，tqdm 的 \r 也能及时读到）。"""
        fd = self.proc.stdout.fileno()
        buf = b''
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                self._emit(line)
        if buf:
            self._emit(buf)
        self.log_queue.put(None)      # 结束哨兵

    def _emit(self, raw_line):
        # Windows 文本模式会把 \n 转成 \r\n：行尾自带的 \r 要先剥离；
        # tqdm 进度条用 \r 刷新：只取最后一个非空片段，避免刷屏
        line = raw_line.decode('utf-8', errors='replace').rstrip('\r\n')
        parts = [p for p in line.split('\r') if p.strip()]
        line = parts[-1] if parts else ''
        if line.strip():
            self.log_queue.put((line + '\n', False))

    def stop_train(self):
        if self.proc and self.proc.poll() is None:
            self.append_log('[UI] 正在终止训练进程…\n', 'sys')
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.append_log('[UI] 进程未响应，已强制结束。\n', 'stderr')

    def _on_training_end(self):
        code = self.proc.poll() if self.proc else None
        el = time.time() - self.start_time if self.start_time else 0
        if code == 0:
            self.append_log(f'[UI] 训练正常结束，用时 {int(el // 60)} 分 {int(el % 60)} 秒。\n', 'sys')
            self.status_var.set('训练完成')
        else:
            self.append_log(f'[UI] 训练结束（退出码 {code}），用时 {int(el // 60)} 分 {int(el % 60)} 秒。\n',
                            'stderr')
            self.status_var.set(f'训练结束（退出码 {code}）')
        self.proc = None
        self.start_time = None
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        for w in self._input_widgets:
            w.config(state='normal')


if __name__ == '__main__':
    app = TrainUI()
    app.mainloop()
