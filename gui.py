import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from core._01_preprocess import preprocess_image_stack
from core._02_pyramids import build_pyramids_stack
from core._03_sharpness import compute_sharpness_map
from core._04_mask import build_masks, build_raw_masks
from core._05_fusion import fuse_pyramids_and_reconstruct

class FocusStackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Focus Stacking GUI")
        self.root.geometry("850x1000")

        # Use absolute paths based on the script location to ensure folders are found
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_dir, "data")
        self.output_dir = os.path.join(base_dir, "output", "fused_images")
        os.makedirs(self.output_dir, exist_ok=True)

        # Animation state
        self.anim_frames = []
        self.anim_id = None
        self.anim_idx = 0
        self.is_playing = True

        self.create_widgets()

    def create_widgets(self):
        # 1. Image Selection
        frame_select = ttk.LabelFrame(self.root, text="1. Select Image Set")
        frame_select.pack(fill="x", padx=10, pady=5)

        self.folder_var = tk.StringVar()
        self.folder_combo = ttk.Combobox(frame_select, textvariable=self.folder_var, state="readonly")
        self.folder_combo.pack(fill="x", padx=10, pady=10)
        self.refresh_folders()

        # 2. Fusion Settings (Mask Type & Top Layer Fusion)
        frame_settings = ttk.LabelFrame(self.root, text="2. Fusion Settings")
        frame_settings.pack(fill="x", padx=10, pady=5)

        # Mask Type
        ttk.Label(frame_settings, text="Mask Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.mask_var = tk.StringVar(value="Soft")
        frame_mask_opts = ttk.Frame(frame_settings)
        frame_mask_opts.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frame_mask_opts, text="Normalized Soft", variable=self.mask_var, value="Soft").pack(side="left", padx=5)
        ttk.Radiobutton(frame_mask_opts, text="Hard", variable=self.mask_var, value="Hard").pack(side="left", padx=5)

        # Top Layer Fusion Method
        ttk.Label(frame_settings, text="Top Layer Fusion:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.top_fusion_var = tk.StringVar(value="max")
        frame_top_opts = ttk.Frame(frame_settings)
        frame_top_opts.grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(frame_top_opts, text="Max", variable=self.top_fusion_var, value="max").pack(side="left", padx=5)
        ttk.Radiobutton(frame_top_opts, text="Mean", variable=self.top_fusion_var, value="mean").pack(side="left", padx=5)

        # 3. Pyramid Levels
        frame_levels = ttk.LabelFrame(self.root, text="3. Pyramid Levels")
        frame_levels.pack(fill="x", padx=10, pady=5)

        self.level_var = tk.IntVar(value=5)
        self.level_label = ttk.Label(frame_levels, text="Levels: 5")
        self.level_label.pack(pady=5)
        
        self.level_scale = ttk.Scale(frame_levels, from_=2, to=20, variable=self.level_var, orient="horizontal", command=self.update_level_label)
        self.level_scale.pack(fill="x", padx=10, pady=10)

        # 4. Generate Button & Progress
        frame_action = ttk.Frame(self.root)
        frame_action.pack(fill="x", padx=10, pady=10)

        self.btn_generate = ttk.Button(frame_action, text="Generate Fused Image", command=self.start_generation)
        self.btn_generate.pack(fill="x", pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame_action, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)

        self.status_label = ttk.Label(frame_action, text="Ready")
        self.status_label.pack()

        # 5. Image Display Area
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Configure grid layout
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(1, weight=1)

        # Titles
        self.lbl_source_title = ttk.Label(self.display_frame, text="Source Images", font=("Arial", 12))
        self.lbl_source_title.grid(row=0, column=0, pady=5)

        self.lbl_result_title = ttk.Label(self.display_frame, text="Fused Result", font=("Arial", 12))
        self.lbl_result_title.grid(row=0, column=1, pady=5)

        # Images
        self.anim_label = ttk.Label(self.display_frame, text="", anchor="center")
        self.anim_label.grid(row=1, column=0, sticky="nsew", padx=5)
        
        self.result_label = ttk.Label(self.display_frame, text="", anchor="center")
        self.result_label.grid(row=1, column=1, sticky="nsew", padx=5)
        
        # Controls (Left side)
        self.controls_frame = ttk.Frame(self.display_frame)
        self.controls_frame.grid(row=2, column=0, sticky="ew", pady=5, padx=5)
        
        self.btn_play = ttk.Button(self.controls_frame, text="Pause", command=self.toggle_play)
        self.btn_play.pack(side="left", padx=5)
        
        self.anim_slider_var = tk.IntVar()
        self.anim_slider = ttk.Scale(self.controls_frame, from_=0, to=0, variable=self.anim_slider_var, orient="horizontal", command=self.on_slider_change)
        self.anim_slider.pack(side="left", fill="x", expand=True, padx=5)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="Pause")
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
                self.anim_id = None
            self.animate_loop()
        else:
            self.btn_play.config(text="Play")
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
                self.anim_id = None

    def on_slider_change(self, value):
        if not self.anim_frames:
            return
        idx = int(float(value))
        self.anim_idx = idx
        img = self.anim_frames[self.anim_idx]
        self.anim_label.config(image=img, text="")

    def refresh_folders(self):
        if os.path.exists(self.data_dir):
            folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
            self.folder_combo['values'] = sorted(folders)
            if folders:
                self.folder_combo.current(0)

    def update_level_label(self, value):
        self.level_label.config(text=f"Levels: {int(float(value))}")

    def start_generation(self):
        folder_name = self.folder_var.get()
        if not folder_name:
            messagebox.showerror("Error", "Please select an image set.")
            return

        self.stop_animation()  # Stop any existing animation
        self.btn_generate.config(state="disabled")
        self.progress_var.set(0)
        self.status_label.config(text="Starting...")
        
        thread = threading.Thread(target=self.run_fusion_pipeline, args=(folder_name,))
        thread.start()

    def run_fusion_pipeline(self, folder_name):
        try:
            levels = int(self.level_var.get())
            mask_type = self.mask_var.get()
            top_method = self.top_fusion_var.get()
            data_path = os.path.join(self.data_dir, folder_name)

            # Step 1: Preprocess
            self.update_status("Preprocessing images...", 10)
            images = preprocess_image_stack(data_path)
            
            # Step 2: Build Pyramids
            self.update_status("Building pyramids...", 30)
            gaussian_pyrs, laplacian_pyrs, top_gaussians = build_pyramids_stack(images, levels)

            # Step 3: Compute Sharpness
            self.update_status("Computing sharpness maps...", 50)
            sharpness_maps = compute_sharpness_map(laplacian_pyrs)

            # Step 4: Build Masks
            self.update_status(f"Building {mask_type} masks...", 70)
            if mask_type == "Soft":
                masks = build_masks(sharpness_maps, sigma=1.2, ksize=7)
            else:
                masks = build_raw_masks(sharpness_maps)

            # Step 5: Fusion
            self.update_status(f"Fusing images (Top: {top_method})...", 90)
            fused_image = fuse_pyramids_and_reconstruct(laplacian_pyrs, top_gaussians, masks, top_fusion_method=top_method)

            # Save
            output_path = os.path.join(self.output_dir, f"{folder_name}_{mask_type}_{top_method}_L{levels}_fused.png")
            cv2.imwrite(output_path, fused_image.astype(np.uint8))
            
            self.update_status("Done!", 100)
            self.root.after(0, lambda: self.show_result(output_path, images))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.update_status("Error occurred", 0)
        finally:
            self.root.after(0, lambda: self.btn_generate.config(state="normal"))

    def update_status(self, text, progress):
        self.root.after(0, lambda: self.status_label.config(text=text))
        self.root.after(0, lambda: self.progress_var.set(progress))

    def show_result(self, image_path, source_images):
        # 1. Show Fused Image (Right)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit half window width roughly
        h, w = img.shape[:2]
        max_h = 450
        max_w = 425
        
        scale = min(max_h / h, max_w / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.result_label.config(image=img_tk, text="")
        self.result_label.image = img_tk

        # 2. Prepare Animation (Left)
        self.anim_frames = []
        for i in range(source_images.shape[0]):
            frame = source_images[i].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (new_w, new_h)) # Match size
            self.anim_frames.append(ImageTk.PhotoImage(Image.fromarray(frame_resized)))
            
        # Initialize controls
        self.anim_slider.config(to=len(self.anim_frames)-1)
        self.anim_slider_var.set(0)
        self.is_playing = True
        self.btn_play.config(text="Pause")
            
        self.start_animation()

    def start_animation(self):
        self.anim_idx = 0
        self.animate_loop()
        
    def animate_loop(self):
        if not self.anim_frames:
            return
        
        if self.is_playing:
            img = self.anim_frames[self.anim_idx]
            self.anim_label.config(image=img, text="")
            self.anim_slider_var.set(self.anim_idx)
            self.anim_idx = (self.anim_idx + 1) % len(self.anim_frames)
            
            # Loop at 10 FPS (100ms)
            self.anim_id = self.root.after(100, self.animate_loop)
        
    def stop_animation(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        self.anim_frames = []
        self.anim_label.config(image="")
        self.result_label.config(image="")

if __name__ == "__main__":
    root = tk.Tk()
    app = FocusStackingGUI(root)
    root.mainloop()
