"""
Image Processing Desktop Application (VisionLab)
================================================
Team Project — Python + OpenCV + Tkinter
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 
import numpy as np
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
#  Color Theme & Fonts
# ─────────────────────────────────────────────
BG, PANEL_BG, CARD_BG = "#343434", "#2a2a2a", "#1f1f1f"
ACCENT, TEXT_LIGHT, TEXT_DIM = "#c1ff72", "#ffffff", "#aaaaaa"
F_TITLE = ("Segoe UI", 20, "bold")
F_BODY  = ("Segoe UI", 11)
F_SMALL = ("Segoe UI", 10)

# ─────────────────────────────────────────────
#  Main Application Class
# ─────────────────────────────────────────────
class ImageProcessorApp:
    """
    Main application class.
    Holds the loaded image, current display image, and all UI widgets.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VisionLab")
        self.root.configure(bg=BG)
        self.root.geometry("1200x740")
        self.root.minsize(900, 600)

        # Image state variables
        self.orig_img = None  # Original loaded image (never modified)
        self.curr_img = None  # Current image after applying filters
        self.refs = []        # To keep PhotoImage references in memory to prevent garbage collection
        
        # Variables linked to sliders
        self.blur_var = tk.IntVar(value=15)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.zoom_var = tk.DoubleVar(value=1.5)
        
        # Build UI and initialize scaling/scrolling
        self._build_ui()
        self.root.update()
        self.cvs.configure(scrollregion=self.cvs.bbox("all"))

    # ─────────────────────────────────────────────
    #  UI Construction Methods
    # ─────────────────────────────────────────────
    def _build_ui(self):
        """Builds the main layout of the application."""
        
        # 1. Top Bar (Header)
        bar = tk.Frame(self.root, bg=CARD_BG, height=60)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡ VisionLab", font=F_TITLE, bg=CARD_BG, fg=ACCENT).pack(side=tk.LEFT, padx=20)
        
        # Action Buttons (Open, Save, Reset)
        right = tk.Frame(bar, bg=CARD_BG)
        right.pack(side=tk.RIGHT, padx=16)
        self._btn(right, "📂 Open Image", self.open_image, True).pack(side=tk.LEFT, padx=5)
        self._btn(right, "💾 Save Image", self.save_image).pack(side=tk.LEFT, padx=5)
        self._btn(right, "↺ Reset", self.reset_img).pack(side=tk.LEFT, padx=5)

        # 2. Status Bar (At the bottom)
        self.status = tk.StringVar(value="No image loaded")
        status_bar = tk.Frame(self.root, bg=BG)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=5)
        tk.Label(status_bar, textvariable=self.status, font=F_SMALL, bg=BG, fg=TEXT_DIM).pack(side=tk.LEFT)

        # 3. Main Body Container
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 5))

        # 4. Sidebar (Scrollable area for operations)
        outer = tk.Frame(body, bg=PANEL_BG, width=280)
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        outer.pack_propagate(False)

        self.cvs = tk.Canvas(outer, bg=PANEL_BG, highlightthickness=0)
        scr = tk.Scrollbar(outer, orient=tk.VERTICAL, command=self.cvs.yview)
        self.cvs.configure(yscrollcommand=scr.set)
        scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.cvs, bg=PANEL_BG)
        win_id = self.cvs.create_window((0, 0), window=self.sidebar, anchor="nw")

        # Bind events to adjust sidebar scrolling dynamically
        self.sidebar.bind("<Configure>", lambda e: self.cvs.configure(scrollregion=self.cvs.bbox("all")))
        self.cvs.bind("<Configure>", lambda e: self.cvs.itemconfig(win_id, width=e.width))
        self.cvs.bind_all("<MouseWheel>", lambda e: self.cvs.yview_scroll(-1*(e.delta//120), "units"))

        self._populate_sidebar()

        # 5. Canvas Area (Main display for the image)
        right_frame = tk.Frame(body, bg=CARD_BG)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(right_frame, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _populate_sidebar(self):
        """Generates all sidebar buttons and sliders dynamically from a list to reduce code duplication."""
        sections = [
            ("Channels", [("🔴 Red", lambda: self.ch("R")), ("🟢 Green", lambda: self.ch("G")), ("🔵 Blue", lambda: self.ch("B"))]),
            ("Grayscale", [("⬛ To Grayscale", self.to_gray)]),
            ("Geometric", [
                ("slider", "Zoom Factor", self.zoom_var, 1.1, 4.0, 0.1),
                ("🔍 Zoom (NN)", lambda: self.zoom("nn")), ("🔎 Zoom (Bilinear)", lambda: self.zoom("bil")),
                ("↻ Rotate CW", lambda: self.rot("cw")), ("↺ Rotate CCW", lambda: self.rot("ccw"))
            ]),
            ("Flip", [("⇆ Horizontal", lambda: self.flip(1)), ("⇅ Vertical", lambda: self.flip(0))]),
            ("Enhancement", [
                ("📊 Hist. Equalization (Compare)", self.hist_eq),
                ("slider", "Gamma", self.gamma_var, 0.1, 3.0, 0.1),
                ("☀️ Apply Gamma", self.apply_gamma)
            ]),
            ("Blur & Edges", [
                ("slider", "Blur Strength", self.blur_var, 1, 51, 2),
                ("🌫 Gaussian Blur", self.apply_blur), ("🔲 Canny Edges", self.edges)
            ]),
            ("Compare", [("📈 Original Histogram", self.compare_view)])
        ]

        tk.Label(self.sidebar, text="OPERATIONS", font=F_SMALL, bg=PANEL_BG, fg=TEXT_DIM).pack(pady=10, padx=16, anchor="w")

        for title, items in sections:
            tk.Label(self.sidebar, text=title.upper(), font=F_SMALL, bg=PANEL_BG, fg=TEXT_DIM).pack(pady=(10,2), padx=16, anchor="w")
            for item in items:
                if item[0] == "slider":
                    # Create a slider with its label
                    f = tk.Frame(self.sidebar, bg=PANEL_BG)
                    f.pack(fill=tk.X, padx=12, pady=2)
                    tk.Label(f, text=item[1], font=F_SMALL, bg=PANEL_BG, fg=TEXT_DIM).pack(anchor="w")
                    tk.Scale(f, from_=item[3], to=item[4], variable=item[2], orient=tk.HORIZONTAL, resolution=item[5], bg=PANEL_BG, fg=TEXT_LIGHT, highlightthickness=0).pack(fill=tk.X)
                else:
                    # Create a standard button
                    self._btn(self.sidebar, item[0], item[1]).pack(pady=2, padx=12, fill=tk.X)

    def _btn(self, parent, text, cmd, accent=False):
        """Creates a custom styled button using Tkinter Labels for better aesthetics."""
        bg, fg = (ACCENT, "#1f1f1f") if accent else ("#404040", TEXT_LIGHT)
        b = tk.Label(parent, text=text, font=F_BODY, bg=bg, fg=fg, cursor="hand2", padx=10, pady=6, anchor="center" if accent else "w")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.config(bg=ACCENT, fg="#1f1f1f"))
        b.bind("<Leave>", lambda e: b.config(bg=bg, fg=fg))
        return b

    # ─────────────────────────────────────────────
    #  Display & Helpers Methods
    # ─────────────────────────────────────────────
    def _on_canvas_resize(self, event=None):
        """Handles resizing of the canvas. Redraws the image or the placeholder text."""
        if self.curr_img is not None:
            self._display(self.curr_img)
        else:
            self._show_placeholder()

    def _show_placeholder(self):
        """Draws the default text when no image is loaded."""
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 700
        ch = self.canvas.winfo_height() or 500
        self.canvas.create_text(cw//2, ch//2, text="📷\nOpen an image to get started", font=("Segoe UI", 14), fill=TEXT_DIM, justify=tk.CENTER)

    def _display(self, img):
        """Converts an OpenCV image to PIL, then to PhotoImage, and draws it on the canvas."""
        if img is None: return
        p_img = Image.fromarray(img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cw, ch = self.canvas.winfo_width() or 700, self.canvas.winfo_height() or 500
        self.photo = ImageTk.PhotoImage(p_img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.photo)
        
        # Update status bar
        ch_count = 1 if len(img.shape)==2 else 3
        self.status.set(f"Size: {img.shape[1]}x{img.shape[0]} | Channels: {ch_count}")

    def _get_hist(self, img, w=400, h=250):
        """Generates a Bar Chart style Histogram for the given image."""
        cvs = np.ones((h, w, 3), dtype=np.uint8) * 30
        channels = [img] if len(img.shape)==2 else cv2.split(img)
        colors = [(200,200,200)] if len(img.shape)==2 else [(255,80,80), (80,255,80), (80,80,255)]
        
        # Draw axes
        cv2.line(cvs, (30, h-20), (w-20, h-20), (255,255,255), 1)
        cv2.line(cvs, (30, h-20), (30, 20), (255,255,255), 1)

        # Draw histogram bars
        for ch, col in zip(channels, colors):
            hist = cv2.calcHist([ch], [0], None, [256], [0,256])
            cv2.normalize(hist, hist, 0, h-50, cv2.NORM_MINMAX)
            for x in range(256):
                val = int(hist[x][0])
                if val > 0:
                    px = 30 + int(x * ((w-50) / 256)) 
                    cv2.line(cvs, (px, h-21), (px, h-21-val), col, 1)
        return cvs

    def _get_stats(self, img):
        """Calculates and formats the mean intensity of the image channels."""
        if len(img.shape) == 2:
            m = cv2.mean(img)[0]
            return f"⚪ Grayscale Mean: {m:.2f}"
        else:
            b, g, r = cv2.mean(img)[:3]
            return f"🔴 Red: {r:.1f}   |   🟢 Green: {g:.1f}   |   🔵 Blue: {b:.1f}"

    def _chk(self):
        """Checks if an image is loaded before applying any operation."""
        if self.curr_img is None: 
            messagebox.showwarning("Error", "Please open an image first.")
            return False
        return True

    # ─────────────────────────────────────────────
    #  Core Logic & OpenCV Operations
    # ─────────────────────────────────────────────
    def open_image(self):
        """Opens a file dialog, reads the image, and downscales it if it's too large."""
        p = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        if not p: return
        img = cv2.imread(p)
        
        # Resize if dimensions exceed 800px to fit screen nicely initially
        h, w = img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        self.orig_img = img.copy()
        self.curr_img = self.orig_img.copy()
        self._display(self.curr_img)

    def reset_img(self):
        """Restores the image back to its originally loaded state."""
        if self._chk(): 
            self.curr_img = self.orig_img.copy()
            self._display(self.curr_img)

    def save_image(self):
        """Saves the currently displayed image to the disk."""
        if not self._chk(): return
        p = filedialog.asksaveasfilename(defaultextension=".png")
        if p: 
            cv2.imwrite(p, self.curr_img)
            messagebox.showinfo("Saved", "Image saved successfully!")

    def ch(self, c):
        """Splits the image into BGR channels and displays only the selected one."""
        if self._chk() and len(self.orig_img.shape)==3:
            # zip maps "B", "G", "R" characters to their respective matrices
            self.curr_img = dict(zip("BGR", cv2.split(self.orig_img)))[c]
            self._display(self.curr_img)

    def to_gray(self):
        """Converts the image to Grayscale."""
        if self._chk() and len(self.curr_img.shape)==3:
            self.curr_img = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)
            self._display(self.curr_img)

    def zoom(self, m):
        """Zooms the image using either Nearest Neighbor or Bilinear interpolation."""
        if not self._chk(): return
        f = self.zoom_var.get()
        h, w = self.curr_img.shape[:2]
        
        # Prevent memory overflow crashes
        if w*f > 6000 or h*f > 6000:
            messagebox.showwarning("Memory Limit", "Zooming further will crash the app.")
            return
        
        interp = cv2.INTER_NEAREST if m=="nn" else cv2.INTER_LINEAR
        self.curr_img = cv2.resize(self.curr_img, (int(w*f), int(h*f)), interpolation=interp)
        self._display(self.curr_img)

    def rot(self, d):
        """Rotates the image 90 degrees Clockwise or Counter-Clockwise."""
        if self._chk():
            code = cv2.ROTATE_90_CLOCKWISE if d=="cw" else cv2.ROTATE_90_COUNTERCLOCKWISE
            self.curr_img = cv2.rotate(self.curr_img, code)
            self._display(self.curr_img)

    def flip(self, axis):
        """Flips the image horizontally or vertically."""
        if self._chk():
            self.curr_img = cv2.flip(self.curr_img, axis)
            self._display(self.curr_img)

    def hist_eq(self):
        """
        Applies Histogram Equalization.
        For color images, it converts to YCrCb, equalizes the Y (Luminance) channel, 
        and converts back to BGR to prevent color distortion.
        """
        if not self._chk(): return
        old_img = self.curr_img.copy()
        
        if len(self.curr_img.shape) == 2:
            self.curr_img = cv2.equalizeHist(self.curr_img)
        else:
            yuv = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2YCrCb)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            self.curr_img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
            
        self._display(self.curr_img)
        self._show_compare(old_img, self.curr_img, "Before Histogram Equalization", "After Histogram Equalization", show_graphs=True)

    def apply_gamma(self):
        """Applies Gamma Correction using a Look-Up Table (LUT) for performance."""
        if self._chk():
            g = self.gamma_var.get()
            # Create a mapping of all 256 pixel values to their new gamma-corrected values
            lut = np.array([min(255, int((i/255.0)**(1.0/g)*255)) for i in range(256)], dtype=np.uint8)
            self.curr_img = cv2.LUT(self.curr_img, lut)
            self._display(self.curr_img)

    def apply_blur(self):
        """Applies Gaussian Blur. Ensures the kernel size is an odd number."""
        if self._chk():
            k = self.blur_var.get() | 1 # Bitwise OR ensures 'k' is always odd
            self.curr_img = cv2.GaussianBlur(self.curr_img, (k, k), 0)
            self._display(self.curr_img)

    def edges(self):
        """Applies Canny Edge Detection (requires grayscale conversion first)."""
        if self._chk():
            g = self.curr_img if len(self.curr_img.shape)==2 else cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)
            self.curr_img = cv2.Canny(g, 100, 200)
            self._display(self.curr_img)

    # ─────────────────────────────────────────────
    #  Comparison Window Logic
    # ─────────────────────────────────────────────
    def compare_view(self):
        """Opens the comparison window to show the current image and its histogram."""
        if self._chk():
            self._show_compare(self.curr_img, None, "Output Image", "Histogram Graph", show_graphs=False)

    def _show_compare(self, img1, img2, t1, t2, show_graphs=False):
        """
        Builds a separate Toplevel window to display images and histograms side-by-side.
        Uses grid layout with weights to keep everything centered.
        """
        win = tk.Toplevel(self.root)
        win.title("Compare Window")
        win.configure(bg=CARD_BG)
        win.geometry("1100x850") 
        
        # Configure grid to expand and center content
        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)
        
        tk.Label(win, text=t1, bg=CARD_BG, fg=TEXT_LIGHT, font=F_BODY).grid(row=0, column=0, pady=(20, 5))
        tk.Label(win, text=t2, bg=CARD_BG, fg=TEXT_LIGHT, font=F_BODY).grid(row=0, column=1, pady=(20, 5))

        def add_img(img, r, c, size=(450,350)):
            """Helper to process and place an image in the grid."""
            p = Image.fromarray(img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            p.thumbnail(size, Image.LANCZOS)
            ref = ImageTk.PhotoImage(p)
            self.refs.append(ref) # Keep reference alive
            tk.Label(win, image=ref, bg=CARD_BG).grid(row=r, column=c, padx=10, pady=10)

        add_img(img1, 1, 0)

        if show_graphs:
            # Mode 1: Histogram Equalization (Before & After with Stats)
            add_img(img2, 1, 1) 
            
            tk.Label(win, text="Histogram Graph (Before)", bg=CARD_BG, fg=TEXT_DIM, font=F_SMALL).grid(row=2, column=0, pady=(10,0))
            tk.Label(win, text="Histogram Graph (After)", bg=CARD_BG, fg=TEXT_DIM, font=F_SMALL).grid(row=2, column=1, pady=(10,0))
            
            add_img(self._get_hist(img1, 450, 250), 3, 0)
            add_img(self._get_hist(img2, 450, 250), 3, 1)
            
            tk.Label(win, text=self._get_stats(img1), bg=CARD_BG, fg=ACCENT, font=F_SMALL).grid(row=4, column=0, pady=5)
            tk.Label(win, text=self._get_stats(img2), bg=CARD_BG, fg=ACCENT, font=F_SMALL).grid(row=4, column=1, pady=5)
        else:
            # Mode 2: Original Compare (Image + Histogram with Stats)
            add_img(self._get_hist(img1, 450, 350), 1, 1)
            tk.Label(win, text=self._get_stats(img1), bg=CARD_BG, fg=ACCENT, font=F_BODY).grid(row=2, column=1, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()