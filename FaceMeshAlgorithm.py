import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np

class FaceMeshDetectionFromRealVideo:
    def __init__(self, window, window_title = "Face Mesh Detection Project",video_source = 0):  
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        self.vid = cv2.VideoCapture(video_source)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        self.btn_network = tk.Button(window, text="Network", width=10, command=self.toggle_network)
        self.btn_network.pack(side=tk.LEFT,padx=10, pady=10)
        self.btn_dot = tk.Button(window, text="Dot", width=10, command=self.toggle_dot)
        self.btn_dot.pack(side=tk.LEFT,padx=10, pady=0)
        self.btn_close = tk.Button(window, text="Close", width=10, command=self.close_window)
        self.btn_close.pack(side=tk.LEFT,padx=10, pady=0)
        
        self.is_network = False
        self.is_dot = False
        
        
        
        self.update()
        self.window.mainloop()
    
    """
    below two functions provide change status ( network or dot ) 
    """
    def toggle_network(self):
        self.is_network = not self.is_network
        if self.is_dot :
            self.is_dot = not self.is_dot
            
    def toggle_dot(self):
        self.is_dot = not self.is_dot
        if self.is_network : 
            self.is_network = not self.is_network
    """
    correctly terminates the program
    """
    def close_window(self):
        self.vid.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        self.window.destroy()
    
    """
    change image type for display
    """
    def convert_to_photo(self, frame):
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=img)
        return photo
    
    """
    receives, processes and displays videos depending on the status
    """
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame)
            if results.multi_face_landmarks:
                if self.is_network:
                    for face_landmarks in results.multi_face_landmarks:
                        
                        landmarks_np = np.zeros((len(face_landmarks.landmark), 3), dtype=np.float32)
                        for i, landmark in enumerate(face_landmarks.landmark):
                            landmarks_np[i] = [landmark.x, landmark.y, landmark.z]


                        self.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        
                elif self.is_dot :       
                    for face_landmarks in results.multi_face_landmarks:

                        for id, landmark in enumerate(face_landmarks.landmark):
                            h, w, c = frame.shape
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
            self.photo = self.convert_to_photo(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.window.after(10, self.update)

        
        
class FaceMeshDetectionFromImage:
    def __init__(self, window, source, window_title="Face Mesh Detection Project"):
        
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        self.image = cv2.imread(source)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

        self.canvas = tk.Canvas(window, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas.pack()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.btn_network = tk.Button(window, text="Network", width=10, command=self.toggle_network)
        self.btn_network.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_dot = tk.Button(window, text="Dot", width=10, command=self.toggle_dot)
        self.btn_dot.pack(side=tk.LEFT, padx=10, pady=0)
        self.btn_close = tk.Button(window, text="Close", width=10, command=self.close_window)
        self.btn_close.pack(side=tk.LEFT, padx=10, pady=0)

        self.is_network = False
        self.is_dot = False

        self.process_image()
        self.window.mainloop()
    
    """
    below two functions provide change status ( network or dot ) 
    """
    def toggle_network(self):
        self.is_network = not self.is_network
        if self.is_dot :
            self.is_dot = not self.is_dot
        self.process_image()

    def toggle_dot(self):
        self.is_dot = not self.is_dot
        if self.is_network :
            self.is_network = not self.is_network
        self.process_image()
        
    """
    change image type for display
    """
    def show_image(self):
        image = Image.fromarray(self.image_copy)
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    """
    correctly terminates the program
    """
    def close_window(self):
        cv2.destroyAllWindows()
        self.window.destroy()
    
    """
    receives, processes and displays images depending on the status
    """
    def process_image(self):
        self.image_copy = np.copy(self.image)
        results = self.face_mesh.process(self.image_copy)

        if results.multi_face_landmarks:
                if self.is_dot:
                    for face_landmarks in results.multi_face_landmarks:
                        for id, landmark in enumerate(face_landmarks.landmark):
                            h, w, c =  self.image_copy.shape
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle( self.image_copy, (x, y), 1, (0, 255, 0), -1)
                
                
                elif self.is_network:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_np = np.zeros((len(face_landmarks.landmark), 3), dtype=np.float32)
                        for i, landmark in enumerate(face_landmarks.landmark):
                            landmarks_np[i] = [landmark.x, landmark.y, landmark.z]

                        self.mp_drawing.draw_landmarks(
                            self.image_copy,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())


        self.show_image()

        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMeshDetectionFromRealVideo(root) # video_source ="./*.mp4" default ( 0 )
    #app = FaceMeshDetectionFromImage(root, source="./*.jpg")
