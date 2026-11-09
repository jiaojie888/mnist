
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw

from model_loaders.knn import KNNClassifier
from model_loaders.logistic import LogisticRegression
from model_loaders.mlp import MLPClassifier


class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字识别")
        self.root.geometry("680x460")
        self.root.minsize(580, 400)
        self.root.configure(bg="#f5f6fa")

        self.bg = "#f5f6fa"
        self.blue = "#3478f6"
        self.canvas_size = 224

        # =========================
        # 加载模型
        # =========================
        self.models = {
            "KNN": KNNClassifier(k=5),
            "逻辑回归": LogisticRegression(),
            "MLP": MLPClassifier()
        }

        for model in self.models.values():
            model.load()

        self.selected_models = {
            name: tk.BooleanVar(value=True)
            for name in self.models
        }

        # =========================
        # 主体区域
        # =========================
        main_frame = tk.Frame(root, bg=self.bg)
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=16
        )

        # =========================
        # 左侧：模型选择
        # =========================
        left_frame = tk.Frame(
            main_frame,
            bg=self.bg,
            width=125
        )
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        left_content = tk.Frame(
            left_frame,
            bg=self.bg
        )
        left_content.place(
            relx=0.5,
            rely=0.5,
            anchor="center"
        )

        tk.Label(
            left_content,
            text="选择模型",
            font=("Microsoft YaHei", 10, "bold"),
            bg=self.bg,
            fg="#333333"
        ).pack(pady=(0, 10))

        self.model_buttons = {}

        for name in self.models:
            button = tk.Button(
                left_content,
                text=name,
                command=lambda n=name: self.toggle_model(n),
                font=("Microsoft YaHei", 9, "bold"),
                width=11,
                bd=0,
                relief=tk.FLAT,
                cursor="hand2",
                pady=7
            )
            button.pack(pady=4)
            self.model_buttons[name] = button

        self.update_model_buttons()

        # =========================
        # 中间：画板
        # =========================
        canvas_frame = tk.Frame(
            main_frame,
            bg=self.bg
        )
        canvas_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#111111",
            cursor="cross",
            highlightthickness=0
        )
        self.canvas.place(
            relx=0.5,
            rely=0.5,
            anchor="center"
        )

        self.image = Image.new(
            "L",
            (self.canvas_size, self.canvas_size),
            0
        )
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # =========================
        # 右侧：操作按钮
        # =========================
        right_frame = tk.Frame(
            main_frame,
            bg=self.bg,
            width=125
        )
        right_frame.pack(side=tk.LEFT, fill=tk.Y)
        right_frame.pack_propagate(False)

        right_content = tk.Frame(
            right_frame,
            bg=self.bg
        )
        right_content.place(
            relx=0.5,
            rely=0.5,
            anchor="center"
        )

        tk.Button(
            right_content,
            text="识别",
            command=self.recognize,
            font=("Microsoft YaHei", 10, "bold"),
            width=10,
            bg=self.blue,
            fg="white",
            activebackground="#2463d4",
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            pady=8
        ).pack(pady=5)

        tk.Button(
            right_content,
            text="清空",
            command=self.clear,
            font=("Microsoft YaHei", 10),
            width=10,
            bg="#e4e7ec",
            fg="#333333",
            activebackground="#d5d9e0",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            pady=8
        ).pack(pady=5)

        # =========================
        # 底部：识别结果
        # =========================
        result_frame = tk.Frame(
            root,
            bg=self.bg
        )
        result_frame.pack(
            fill=tk.X,
            padx=12,
            pady=(4, 12)
        )

        self.result_labels = {}

        for name in self.models:
            card = tk.Frame(
                result_frame,
                bg="white",
                highlightthickness=1,
                highlightbackground="#e5e7eb"
            )

            card.pack(
                side=tk.LEFT,
                fill=tk.BOTH,
                expand=True,
                padx=4,
                ipady=5
            )

            tk.Label(
                card,
                text=name,
                font=("Microsoft YaHei", 9, "bold"),
                bg="white",
                fg="#555555"
            ).pack(pady=(6, 2))

            prediction_label = tk.Label(
                card,
                text="--",
                font=("Arial", 28, "bold"),
                bg="white",
                fg=self.blue
            )
            prediction_label.pack()

            confidence_label = tk.Label(
                card,
                text="等待识别",
                font=("Microsoft YaHei", 9),
                bg="white",
                fg="#888888"
            )
            confidence_label.pack(pady=(2, 8))

            self.result_labels[name] = (
                prediction_label,
                confidence_label
            )

    # =========================
    # 模型选择
    # =========================
    def toggle_model(self, name):
        var = self.selected_models[name]
        var.set(not var.get())
        self.update_model_buttons()

    def update_model_buttons(self):
        for name, button in self.model_buttons.items():
            if self.selected_models[name].get():
                button.config(
                    text=f"✓  {name}",
                    bg=self.blue,
                    fg="white",
                    activebackground="#2463d4",
                    activeforeground="white"
                )
            else:
                button.config(
                    text=f"○  {name}",
                    bg="white",
                    fg="#333333",
                    activebackground="#eeeeee",
                    activeforeground="#333333"
                )

    # =========================
    # 绘制数字
    # =========================
    def paint(self, event):
        x, y = event.x, event.y
        r = 8

        self.canvas.create_oval(
            x-r, y-r,
            x+r, y+r,
            fill="white",
            outline="white"
        )

        self.draw.ellipse(
            (x-r, y-r, x+r, y+r),
            fill=255
        )

    # =========================
    # 识别数字
    # =========================
    def recognize(self):
        selected = [
            name
            for name, var in self.selected_models.items()
            if var.get()
        ]

        if not selected:
            return

        image = self.image.resize((28, 28))
        image = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        for name, model in self.models.items():
            prediction_label, confidence_label = (
                self.result_labels[name]
            )

            if name not in selected:
                prediction_label.config(text="--")
                confidence_label.config(text="未选择")
                continue

            try:
                prediction, confidence = model.predict(image)

                prediction_label.config(
                    text=str(prediction),
                    fg=self.blue
                )

                confidence_label.config(
                    text=f"置信度 {confidence:.1%}"
                )

            except Exception as e:
                prediction_label.config(
                    text="错误",
                    fg="#dc2626"
                )
                confidence_label.config(text=str(e))

    # =========================
    # 清空画板
    # =========================
    def clear(self):
        self.canvas.delete("all")

        self.image = Image.new(
            "L",
            (self.canvas_size, self.canvas_size),
            0
        )
        self.draw = ImageDraw.Draw(self.image)

        for prediction_label, confidence_label in (
            self.result_labels.values()
        ):
            prediction_label.config(text="--")
            confidence_label.config(text="等待识别")


if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTApp(root)
    root.mainloop()
