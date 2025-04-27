import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from deepface import DeepFace
from src.moodMusicCurator.logging import MoodMusicLogger

logger = MoodMusicLogger.get_logger()

class FacialAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mood Music Curator - Facial Analysis")
        self.root.geometry("800x600")

        # Create GUI elements
        self.label = tk.Label(self.root, text="Select an option:", font=("Arial", 16))
        self.label.pack(pady=10)

        self.webcam_button = tk.Button(self.root, text="Use Webcam", command=self.use_webcam, width=20)
        self.webcam_button.pack(pady=5)

        self.file_button = tk.Button(self.root, text="Select Image File", command=self.select_file, width=20)
        self.file_button.pack(pady=5)

        self.preview_label = tk.Label(self.root)
        self.preview_label.pack(pady=10)

        self.capture_button = tk.Button(self.root, text="Capture Photo (Press 'c')", command=self.capture_photo, width=20)
        self.capture_button.pack(pady=5)
        self.capture_button.config(state=tk.DISABLED)

        # Initialize webcam
        self.cap = None
        self.photo = None
        self.webcam_active = False

    def use_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Error: Could not open webcam.")
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.webcam_active = True
        self.capture_button.config(state=tk.NORMAL)
        self.update_webcam_preview()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.analyze_image(file_path)
        else:
            logger.info("No file selected.")

    def update_webcam_preview(self):
        if self.webcam_active and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)
                self.preview_label.config(image=self.photo)
                self.preview_label.image = self.photo
            self.root.after(10, self.update_webcam_preview)

    def capture_photo(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, frame)
                logger.info("Captured photo from webcam.")
                self.webcam_active = False
                self.cap.release()
                self.capture_button.config(state=tk.DISABLED)
                self.preview_label.config(image='')
                self.analyze_image(temp_path)
                os.remove(temp_path)
            else:
                logger.error("Error: Could not capture photo from webcam.")
                messagebox.showerror("Error", "Could not capture photo from webcam.")

    def analyze_image(self, image_path):
        logger.info(f"Analyzing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Error: Could not read image file '{image_path}'.")
            messagebox.showerror("Error", f"Could not read image file '{image_path}'.")
            return

        try:
            faces = DeepFace.extract_faces(img_path=img, detector_backend='retinaface', enforce_detection=False)
            logger.debug(f"Detected {len(faces)} faces in image")
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            messagebox.showerror("Error", f"Error detecting faces: {e}")
            return

        if not faces:
            logger.info(f"No faces detected in {image_path}")
            messagebox.showinfo("Info", "No faces detected in the image.")
            return

        for i, face_data in enumerate(faces):
            facial_area = face_data['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face_img = img[y:y+h, x:x+w]

            try:
                analysis = DeepFace.analyze(img_path=face_img, actions=['age', 'gender', 'emotion', 'race'], detector_backend='retinaface', enforce_detection=False)
                analysis = analysis[0]
                landmarks = face_data.get('landmarks', {})
                log_message = f"Image: {image_path}, Face {i+1}\n"
                log_message += f"Landmarks: {landmarks}\n"
                log_message += f"Age: {analysis['age']}\n"
                log_message += f"Gender: {analysis['dominant_gender']}\n"
                log_message += f"Emotion: {analysis['dominant_emotion']}\n"
                log_message += f"Race: {analysis['dominant_race']}\n"
                logger.info(log_message)
                messagebox.showinfo("Analysis Result", log_message)
            except Exception as e:
                logger.error(f"Error analyzing Face {i+1} in {image_path}: {e}")
                messagebox.showerror("Error", f"Error analyzing Face {i+1}: {e}")

    def __del__(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()