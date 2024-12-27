import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from numpy.linalg import norm
from deepface import DeepFace
from db import *
import ast
import os

def relative_path(filename):
    return os.path.join(os.path.dirname(__file__), 'images', filename)

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别app")
        self.root.geometry("1160x640")

        # 左
        self.left_frame = tk.Frame(root, width=400, height=600, bg="lightgray")
        self.left_frame.grid(row=0, rowspan=2, column=0, padx=10, pady=10, sticky="nsew")

        # 中
        self.middle_frame = tk.Frame(root, width=100, height=600)
        self.middle_frame.grid(row=0, rowspan=2, column=1, padx=10, pady=10, sticky="nsew")

        # 右
        self.right_up_frame = tk.Frame(root, width=300, height=300)
        self.right_up_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.right_down_frame = tk.Frame(root, width=300, height=300)
        self.right_down_frame.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

        #  左: label
        self.image_label = tk.Label(self.left_frame, width=400, height=600, bg="lightgrey")
        self.image_label.pack()
        image = Image.open(relative_path("default.png"))
        image.thumbnail((600, 600))
        self.img = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.img)

        # 中: Listbox
        self.face_listbox = tk.Listbox(self.middle_frame, height=30, width=30)
        self.face_listbox.pack(fill=tk.BOTH, expand=True)

        # 右: button/entry/label
        self.load_button = tk.Button(self.right_up_frame, text="选择照片", command=self.load_image)
        self.load_button.grid(row=0,column=0,columnspan=2,padx=10, pady=10,sticky="we")

        self.recognize_button = tk.Button(self.right_up_frame, text="识别人脸", command=self.recognize_faces)
        self.recognize_button.grid(row=1,column=0,columnspan=2,padx=10, pady=10,sticky="we")

        self.reset_button = tk.Button(self.right_up_frame, text="重置", command=self.reset)
        self.reset_button.grid(row=2,column=0,columnspan=2,padx=10, pady=10,sticky="we")

        tk.Label(self.right_down_frame, text="姓名:").grid(row=3,column=0,padx=10, pady=10,sticky="we")
        self.entry_name = tk.Entry(self.right_down_frame)
        self.entry_name.grid(row=3,column=1,padx=10, pady=10,sticky="we")

        tk.Label(self.right_down_frame, text="年龄:").grid(row=4,column=0,padx=10, pady=10,sticky="we")
        self.entry_age = tk.Entry(self.right_down_frame)
        self.entry_age.grid(row=4,column=1,padx=10, pady=10,sticky="we")

        tk.Label(self.right_down_frame, text="性别:").grid(row=5,column=0,padx=10, pady=10,sticky="we")
        self.entry_gender = tk.Entry(self.right_down_frame)
        self.entry_gender.grid(row=5,column=1,padx=10, pady=10,sticky="we")

        tk.Label(self.right_down_frame, text="手机:").grid(row=6,column=0,padx=10, pady=10,sticky="we")
        self.entry_phone = tk.Entry(self.right_down_frame)
        self.entry_phone.grid(row=6,column=1,padx=10, pady=10,sticky="we")

        tk.Label(self.right_down_frame, text="备注:").grid(row=7,column=0,padx=10, pady=10,sticky="we")
        self.entry_note = tk.Entry(self.right_down_frame)
        self.entry_note.grid(row=7,column=1,padx=10, pady=10,sticky="we")

        self.save_button = tk.Button(self.right_down_frame, text="录入/更新", command=self.save)
        self.save_button.grid(row=8,column=0,columnspan=2,padx=10, pady=10,sticky="we")

        # 比例调整
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=3)

        self.right_up_frame.grid_columnconfigure(0, weight=1)
        
        self.right_down_frame.grid_columnconfigure(0, weight=1)
        self.right_down_frame.grid_columnconfigure(1, weight=1)

        # 一些成员变量
        self.db = FaceDatabase()
        self.image_path = None
        self.faces = []
        self.select_index = -1

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            try:
                image = Image.open(self.image_path)
                image.thumbnail((600, 600))
                self.img = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.img)
                self.faces.clear()
                self.select_index = -1
                self.face_listbox.delete(0, tk.END)
                self.entry_name.delete(0, tk.END)
                self.entry_age.delete(0, tk.END)
                self.entry_gender.delete(0, tk.END)
                self.entry_phone.delete(0, tk.END)
                self.entry_note.delete(0, tk.END)
            except Exception as e:
                messagebox.showerror("加载失败", f"图片加载失败：{str(e)}")

    def recognize_faces(self):
        if not self.image_path:
            messagebox.showwarning("警告", "请先加载图片")
            return
        try:
            self.faces.clear()
            self.select_index = -1
            self.face_listbox.delete(0, tk.END)
            self.entry_name.delete(0, tk.END)
            self.entry_age.delete(0, tk.END)
            self.entry_gender.delete(0, tk.END)
            self.entry_phone.delete(0, tk.END)
            self.entry_note.delete(0, tk.END)
            recorded_faces = self.db.get_all_faces()
            representations = DeepFace.represent(self.image_path, model_name="VGG-Face")
            self.faces = representations
            image = Image.open(self.image_path)
            draw = ImageDraw.Draw(image)

            filtered_faces = []
            for i, face in enumerate(self.faces):
                region = face['facial_area']
                if not region['left_eye'] and not region['right_eye']:
                    continue
                filtered_faces.append(face)
            self.faces = filtered_faces

            for i, face in enumerate(self.faces):
                region = face['facial_area']
                start_point = (region['x'], region['y'])
                end_point = (region['x'] + region['w'], region['y'] + region['h'])
                draw.rectangle([start_point, end_point], outline="red", width=3)
                text_x = region['x']
                text_y = region['y'] - 40
                text = f"face {i + 1}"
                draw.text((text_x, text_y), text, fill="red", font=ImageFont.truetype("arial.ttf", 40))
                # 判断是否是已识别的人
                for recorded_face in recorded_faces:
                    embedding1 = np.array(face["embedding"])
                    embedding2 = np.array(ast.literal_eval(recorded_face[6])).astype(np.float64)
                    distance = 1 - (np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2)))
                    print(distance)
                    if distance < 0.6:
                        self.faces[i]['db_id'] = recorded_face[0]
                        self.faces[i]['db_name'] = recorded_face[1]
                        self.faces[i]['db_gender'] = recorded_face[2]
                        self.faces[i]['db_age'] = recorded_face[3]
                        self.faces[i]['db_phone'] = recorded_face[4]
                        self.faces[i]['db_note'] = recorded_face[5]
                        break
                else:
                    self.faces[i]['db_id'] = -1
                    self.faces[i]['db_name'] = ""
                    self.faces[i]['db_gender'] = ""
                    self.faces[i]['db_age'] = ""
                    self.faces[i]['db_phone'] = ""
                    self.faces[i]['db_note'] = ""
                if self.faces[i]['db_name']:
                    name = self.faces[i]['db_name']
                    self.face_listbox.insert(tk.END, f"face {i + 1}({name})")
                else:
                    self.face_listbox.insert(tk.END, f"face {i + 1}")
            image.thumbnail((600, 600))
            self.img = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.img)
        except Exception as e:
            messagebox.showerror("识别失败", f"人脸识别失败：{str(e)}")

    def reset(self):
        self.image_path = None
        self.faces = []
        image = Image.open(relative_path("default.png"))
        image.thumbnail((600, 600))
        self.img = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.img)
        self.faces.clear()
        self.select_index = -1
        self.face_listbox.delete(0, tk.END)
        self.entry_name.delete(0, tk.END)
        self.entry_age.delete(0, tk.END)
        self.entry_gender.delete(0, tk.END)
        self.entry_phone.delete(0, tk.END)
        self.entry_note.delete(0, tk.END)

    def save(self):
        if self.select_index == -1:
            messagebox.showerror("错误", "先选择face")
            return
        selected_face = self.faces[self.select_index]
        if selected_face["db_id"] == -1:
            # 新增
            embedding = np.array(self.faces[self.select_index]['embedding']).tolist()
            self.db.insert_face(
                name=self.entry_name.get(),
                gender=self.entry_gender.get(),
                age=self.entry_age.get(),
                phone=self.entry_phone.get(),
                notes=self.entry_note.get(),
                embedding=embedding
            )
            messagebox.showinfo("信息","录入成功")
        else:
            # 更新
            db_id = selected_face["db_id"]
            self.db.update_face(db_id, self.entry_name.get(), self.entry_gender.get(), self.entry_age.get(), self.entry_phone.get(), self.entry_note.get(), None)
            messagebox.showinfo("信息","更新成功")

    def select_face(self, event):
        try:
            self.entry_name.delete(0, tk.END)
            self.entry_age.delete(0, tk.END)
            self.entry_gender.delete(0, tk.END)
            self.entry_phone.delete(0, tk.END)
            self.entry_note.delete(0, tk.END)

            self.select_index = self.face_listbox.curselection()[0]
            selected_face = self.faces[self.select_index]
            
            if selected_face['db_id'] != -1:
                self.entry_name.insert(0, selected_face['db_name'])
                self.entry_gender.insert(0, selected_face['db_gender'])
                self.entry_age.insert(0, selected_face['db_age'])
                self.entry_phone.insert(0, selected_face['db_phone'])
                self.entry_note.insert(0, selected_face['db_note'])

        except IndexError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)

    app.face_listbox.bind("<ButtonRelease-1>", app.select_face)

    root.mainloop()
