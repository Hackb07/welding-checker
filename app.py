import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("welding.pt")

# Initialize customtkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🖼️ Welding Photo Detection")
        self.geometry("900x700")

        self.label = ctk.CTkLabel(self, text="🖼️ Welding Photo Detection", font=("Helvetica", 20))
        self.label.pack(pady=10)

        self.image_label = ctk.CTkLabel(self)
        self.image_label.pack(pady=10)

        self.load_button = ctk.CTkButton(self, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        self.detect_button = ctk.CTkButton(self, text="Run Detection", command=self.run_detection)
        self.detect_button.pack(pady=5)
        self.detect_button.configure(state="disabled")  # Disabled until image loaded

        self.original_image = None  # To store loaded image in OpenCV format
        self.annotated_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            # Load image with OpenCV
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                ctk.CTkMessageBox.show_error("Error", "Failed to load image.")
                return
            self.original_image = img_cv

            # Convert to display in tkinter
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((800, 600))
            imgtk = ImageTk.PhotoImage(img_pil)

            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

            self.detect_button.configure(state="normal")

    def run_detection(self):
        if self.original_image is None:
            return

        # Run YOLO detection on original image
        results = model(self.original_image, verbose=False)
        annotated_frame = results[0].plot()

        # Store annotated image
        self.annotated_image = annotated_frame

        # Convert to display
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((800, 600))
        imgtk = ImageTk.PhotoImage(img_pil)

        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

if __name__ == "__main__":
    app = YOLOApp()
    app.mainloop()
