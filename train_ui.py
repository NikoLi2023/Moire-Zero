# -*- coding: utf-8 -*-
"""MZNet 训练 UI：可视化设置训练参数，一键启动 quick_train.py，并把训练过程的
Print 输出实时呈现在界面上。

用法（在仓库根目录、使用训练环境 python 运行）：

    python train_ui.py

说明：
  - 必填：训练/验证数据集目录名、任务名 name、最终权重名 ckpt_name、总轮数 epochs；
  - 选填：学习率 lr（留空沿用 yml）、数据父目录、基础配置、批量 batch_size；
    初始权重不在界面配置，始终使用基础配置 yml 中 training.resume 的设置；
  - 训练结束后始终自动导出 fp16 ONNX 到结果目录（results/<任务名>/），
    文件名为"权重名+.onnx"（即 <ckpt_name>.onnx），无需勾选；
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
from tkinter import filedialog, scrolledtext, ttk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 仓库根目录（quick_train.py 所在处）


class TrainUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MZNet 训练控制台')
        self.geometry('900x640')
        self.minsize(760, 520)

        self.proc = None            # 训练子进程
        self.log_queue = queue.Queue()
        self.start_time = None
        self._input_widgets = []    # 运行时禁用的输入控件

        self._build_params()
        self._build_log()
        self._build_statusbar()

        self.after(100, self._poll_log)

    # ---------------- 界面布局 ----------------
    def _build_params(self):
        # ---------------- 必填项 ----------------
        req = ttk.LabelFrame(self, text='必填项')
        req.pack(fill='x', padx=10, pady=(10, 2))

        self.var_train = tk.StringVar()
        self.var_val = tk.StringVar()
        self.var_name = tk.StringVar()
        self.var_ckpt_name = tk.StringVar()
        self.var_epochs = tk.StringVar()

        self._dataset_boxes = []
        for r, (label, var) in enumerate((('训练数据集 train_dataset', self.var_train),
                                          ('验证数据集 val_dataset', self.var_val),
                                          ('任务名 name', self.var_name),
                                          ('最终权重名 ckpt_name', self.var_ckpt_name),
                                          ('总轮数 epochs', self.var_epochs))):
            ttk.Label(req, text=label + ' *', foreground='#c0392b').grid(
                row=r, column=0, sticky='e', padx=6, pady=3)
            if var in (self.var_train, self.var_val):
                cmb = ttk.Combobox(req, textvariable=var, values=[], width=39)
                cmb.grid(row=r, column=1, sticky='w', padx=4, pady=3)
                self._input_widgets.append(cmb)
                self._dataset_boxes.append(cmb)
            else:
                ent = ttk.Entry(req, textvariable=var, width=42)
                ent.grid(row=r, column=1, sticky='w', padx=4, pady=3)
                self._input_widgets.append(ent)

        # ---------------- 可选项 ----------------
        opt = ttk.LabelFrame(self,
                             text='可选项')
        opt.pack(fill='x', padx=10, pady=2)

        self.var_data_dir = tk.StringVar(value=SCRIPT_DIR)
        self.var_lr = tk.StringVar()
        self.var_config = tk.StringVar(value='GQMoire.yml')
        self.var_batch = tk.StringVar()

        def row(parent, r, label, var, browse=None, width=42):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky='e', padx=6, pady=3)
            ent = ttk.Entry(parent, textvariable=var, width=width)
            ent.grid(row=r, column=1, sticky='w', padx=4, pady=3)
            self._input_widgets.append(ent)
            if browse:
                btn = ttk.Button(parent, text='浏览…', width=8,
                                 command=lambda: self._browse(var, browse))
                btn.grid(row=r, column=2, sticky='w', padx=4)
                self._input_widgets.append(btn)
            return ent

        row(opt, 0, '数据父目录 data_dir', self.var_data_dir, browse='dir')
        self.btn_refresh_ds = ttk.Button(opt, text='刷新数据集列表',
                                         command=self._refresh_datasets, width=16)
        self.btn_refresh_ds.grid(row=0, column=3, sticky='w', padx=4)
        row(opt, 1, '学习率 lr', self.var_lr)
        row(opt, 2, '基础配置 config', self.var_config)
        row(opt, 3, '批量 batch_size', self.var_batch)

        # 初始扫描一次数据集目录
        self.after(200, self._refresh_datasets)

        # 按钮行
        btns = ttk.Frame(self)
        btns.pack(fill='x', padx=10, pady=2)
        self.btn_start = ttk.Button(btns, text='▶ 开始训练', command=self.start_train)
        self.btn_start.pack(side='left', padx=2)
        self.btn_stop = ttk.Button(btns, text='■ 停止', command=self.stop_train, state='disabled')
        self.btn_stop.pack(side='left', padx=2)
        ttk.Button(btns, text='清空日志', command=self.clear_log).pack(side='left', padx=2)
        ttk.Button(btns, text='打开结果目录',
                   command=lambda: os.startfile(os.path.join(SCRIPT_DIR, 'results'))
                   if os.path.isdir(os.path.join(SCRIPT_DIR, 'results')) else None
                   ).pack(side='left', padx=2)

    def _build_log(self):
        frame = ttk.LabelFrame(self, text='训练输出')
        frame.pack(fill='both', expand=True, padx=10, pady=4)
        self.log_text = scrolledtext.ScrolledText(frame, wrap='none', state='normal',
                                                  font=('Consolas', 10))
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)
        self.log_text.tag_config('stderr', foreground='#c0392b')
        self.log_text.tag_config('sys', foreground='#2980b9')

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value='就绪')
        bar = ttk.Frame(self)
        bar.pack(fill='x', side='bottom')
        ttk.Separator(bar, orient='horizontal').pack(fill='x')
        ttk.Label(bar, textvariable=self.status_var).pack(side='left', padx=8, pady=2)

    def _browse(self, var, kind):
        if kind == 'dir':
            p = filedialog.askdirectory(initialdir=SCRIPT_DIR)
        else:
            p = filedialog.askopenfilename(
                initialdir=SCRIPT_DIR,
                filetypes=[('权重文件', '*.pth *.pth.tar'), ('所有文件', '*.*')])
        if p:
            var.set(p)
            if kind == 'dir':
                self._refresh_datasets()

    def _refresh_datasets(self):
        """扫描 <data_dir>/<名字>/train|test 结构，把数据集目录名填进下拉框。"""
        data_dir = self.var_data_dir.get().strip() or SCRIPT_DIR
        names = []
        try:
            for name in sorted(os.listdir(data_dir)):
                sub = os.path.join(data_dir, name)
                if os.path.isdir(sub) and \
                        os.path.isdir(os.path.join(sub, 'train')) and \
                        os.path.isdir(os.path.join(sub, 'test')):
                    names.append(name)
        except OSError:
            pass
        for cmb in self._dataset_boxes:
            cmb.config(values=names)
        self.status_var.set(f'数据父目录 {data_dir} 下发现 {len(names)} 个数据集（含 train/test 结构）'
                            if names else
                            f'数据父目录 {data_dir} 下未发现 train/test 结构的数据集，请手动填写')

    # ---------------- 日志 ----------------
    def append_log(self, text, tag=None):
        self.log_text.insert('end', text, tag)
        self.log_text.see('end')

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
        val = self.var_val.get().strip()
        name = self.var_name.get().strip()
        ckpt_name = self.var_ckpt_name.get().strip()
        lr = self.var_lr.get().strip()
        epochs = self.var_epochs.get().strip()
        if not train:
            return None, '请填写训练数据集 train_dataset'
        if not val:
            return None, '请填写验证数据集 val_dataset'
        if not name:
            return None, '请填写任务名 name'
        if not ckpt_name:
            return None, '请填写最终权重名 ckpt_name'
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
            'val_dataset': val,
            'lr': lr_value,
            'epochs': epochs,
            'name': name,
            'ckpt_name': ckpt_name,
        }, None

    def start_train(self):
        params, err = self._validate()
        if err:
            self.append_log(f'[UI] 参数错误：{err}\n', 'sys')
            return

        cmd = [sys.executable, '-u', 'quick_train.py',
               '--train_dataset', params['train_dataset'],
               '--val_dataset', params['val_dataset'],
               '--epochs', params['epochs'],
               '--name', params['name'],
               '--ckpt_name', params['ckpt_name']]
        if params['lr'] is not None:
            cmd += ['--lr', str(params['lr'])]
        if self.var_data_dir.get().strip():
            cmd += ['--data_dir', self.var_data_dir.get().strip()]
        if self.var_config.get().strip():
            cmd += ['--config', self.var_config.get().strip()]
        if self.var_batch.get().strip():
            cmd += ['--batch_size', self.var_batch.get().strip()]
        # 不传 --resume：初始权重始终使用 config（基础配置 yml）中 training.resume 的设置
        cmd.append('--export_onnx')   # 训练结束后默认始终自动导出 fp16 ONNX

        self.append_log('[UI] 启动训练：' + ' '.join(cmd) + '\n', 'sys')

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
