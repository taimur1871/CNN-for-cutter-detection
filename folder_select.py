#tkinter test
import tkinter as tk
from tkinter import filedialog
import os

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.open_folder = tk.Button(self)
        self.open_folder["text"] = "Select Folder"
        self.open_folder["command"] = self.select_folder
        self.open_folder.pack(side="top")

    def select_folder(self):
        root.directory = filedialog.askdirectory()
        print(root.directory)
        self.master.destroy()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
