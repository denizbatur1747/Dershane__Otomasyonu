import tkinter as tk
from tkinter import messagebox, font as tkFont, Listbox, Scrollbar, Frame
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import pickle
import time

# --- Ayarlar ---
DATA_DIR = "face_data"
TRAINER_DIR = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.pickle")
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml" # Bu dosyayı kodla aynı dizine koyun veya tam yolunu belirtin

FONT_FAMILY = "Century Gothic"
FONT_SIZE = 18
FONT_STYLE = "normal"
CUSTOM_FONT = (FONT_FAMILY, FONT_SIZE, FONT_STYLE)
LISTBOX_FONT = (FONT_FAMILY, FONT_SIZE - 4, FONT_STYLE) # Listbox için biraz daha küçük font
DARK_VIOLET = "#9400D3" # Koyu Menekşe Rengi Hex Kodu

CONFIDENCE_THRESHOLD = 65 # LBPH için Eşik Değeri (Düşük olması daha iyi eşleşme demek, %'ye çevirirken 100-conf yaparız. Bu değeri ayarlamanız gerekebilir)
REQUIRED_REGISTER_IMAGES = 5
REQUIRED_LOGIN_IMAGES = 1
ADMIN_USER = "admin" # Admin kullanıcı adı (küçük harf)

# --- Gerekli Klasörleri Oluştur ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TRAINER_DIR):
    os.makedirs(TRAINER_DIR)

# --- OpenCV Yüz Algılayıcı ve Tanıyıcı ---
if not os.path.exists(HAAR_CASCADE_PATH):
    messagebox.showerror("Hata", f"{HAAR_CASCADE_PATH} bulunamadı. Lütfen dosyayı temin edin.")
    exit()

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
# recognizer'ı global yapmak yerine gerektiğinde oluşturalım veya load_trained_data içinde yönetelim
recognizer = None
labels = {} # ID -> İsim eşleşmesi için

# --- Eğitilmiş Modeli ve Etiketleri Yükle ---
def load_trained_data():
    global labels, recognizer
    if os.path.exists(TRAINER_FILE) and os.path.exists(LABELS_FILE):
        # LBPH tanıyıcıyı burada oluştur ve eğitilmiş veriyi oku
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)
        with open(LABELS_FILE, 'rb') as f:
            labels = pickle.load(f)
        print("Eğitilmiş model ve etiketler yüklendi.")
        return True
    print("Eğitilmiş model bulunamadı.")
    recognizer = None # Model yüklenemezse None yapalım
    return False

# --- Modeli Eğitme Fonksiyonu ---
def train_model():
    global recognizer # Eğitilen modeli global değişkene de atayalım
    current_id = 0
    label_ids = {} # isim -> ID
    face_samples = []
    ids = []
    print("Yüz verileri taranıyor ve model eğitiliyor...")

    base_dir = DATA_DIR
    for root_dir, dirs, files in os.walk(base_dir):
        for dir_name in dirs: # Her bir kullanıcı klasörü
            user_name = dir_name
            user_label = user_name # Etiket olarak klasör adını (kullanıcı adını) kullanıyoruz

            if user_label not in label_ids:
                label_ids[user_label] = current_id
                current_id += 1
            user_id = label_ids[user_label]

            user_dir_path = os.path.join(root_dir, dir_name)
            print(f"İşlenen klasör: {user_dir_path}, Kullanıcı ID: {user_id}")

            for filename in os.listdir(user_dir_path):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(user_dir_path, filename)
                    try:
                        pil_image = Image.open(img_path).convert("L")
                        image_array = np.array(pil_image, "uint8")
                        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)

                        # Eğitim için tüm gri resmi veya sadece yüzü kullanabilirsiniz.
                        # Kayıt sırasında kırpılmışsa tüm resmi kullanmak yeterli.
                        face_samples.append(image_array)
                        ids.append(user_id)
                        # print(f"  -> {filename} eklendi. ID: {user_id}") # Çok fazla log üretebilir

                    except Exception as e:
                        print(f"Hata: {img_path} işlenemedi - {e}")

    if not face_samples or not ids:
        print("Eğitilecek yüz verisi bulunamadı.")
        return False

    print("Veriler hazırlandı. Model eğitiliyor...")
    id_to_label = {v: k for k, v in label_ids.items()}
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(id_to_label, f)
    print(f"Etiketler kaydedildi: {id_to_label}")

    # Tanıyıcıyı oluştur ve eğit
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_FILE) # Eğitilmiş modeli kaydet
    print(f"Model eğitildi ve '{TRAINER_FILE}' olarak kaydedildi.")
    load_trained_data() # Yeni eğitilen modeli hemen yükle (global recognizer ve labels güncellenir)
    return True


class FaceCaptureApp:
    def __init__(self, parent_window, mode, user_name, required_images):
        self.parent_window = parent_window
        self.mode = mode  # 'register' or 'login'
        self.user_name = user_name
        self.required_images = required_images
        self.captured_images = 0
        self.is_running = False
        self.detected_name = "Bilinmiyor"
        self.confidence_score = 0
        self.login_handled = False  # Giriş durumu işlendi mi kontrolü

        self.capture_window = tk.Toplevel(parent_window)
        self.capture_window.title("Yüz Tanıma")
        self.capture_window.configure(bg="white")
        self.capture_window.attributes("-alpha", 1.0)

        parent_window.update_idletasks()
        parent_x = parent_window.winfo_x()
        parent_y = parent_window.winfo_y()
        parent_width = parent_window.winfo_width()
        parent_height = parent_window.winfo_height()
        win_width = 660
        win_height = 580
        x = parent_x + (parent_width // 2) - (win_width // 2)
        y = parent_y + (parent_height // 2) - (win_height // 2)
        self.capture_window.geometry(f'{win_width}x{win_height}+{x}+{y}')
        self.capture_window.resizable(False, False)
        self.capture_window.transient(parent_window)
        self.capture_window.grab_set()

        self.video_label = tk.Label(self.capture_window, bg="black")
        self.video_label.pack(pady=10)

        self.info_label = tk.Label(self.capture_window, text="Kameraya bakın...", font=CUSTOM_FONT, fg=DARK_VIOLET, bg="white")
        self.info_label.pack(pady=10)

        self.progress_label = tk.Label(self.capture_window, text="", font=CUSTOM_FONT, bg="white")
        self.progress_label.pack(pady=5)

        self.result_label = tk.Label(self.capture_window, text="", font=CUSTOM_FONT, fg="blue", bg="white")
        self.result_label.pack(pady=5)

        self.cancel_button = tk.Button(self.capture_window, text="İptal", command=self.stop_capture, font=CUSTOM_FONT, bg="red", fg="white")
        self.cancel_button.pack(pady=10)

        self.capture_window.protocol("WM_DELETE_WINDOW", self.stop_capture)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Kamera Hatası", "Kamera açılamadı!", parent=self.capture_window)
            self.cleanup()
            return

        if self.mode == 'login' and recognizer is None:
            if not load_trained_data():
                messagebox.showerror("Hata", "Giriş yapılamıyor. Eğitilmiş model bulunamadı.", parent=self.capture_window)
                self.cleanup()
                return

        # --- KAMERA AÇILDI MESAJI ---
        self.info_label.config(text="Kamera açılıyor, lütfen bekleyin...")
        self.capture_window.after(1000, self.start_capture)  # 1 saniye sonra başla

    def start_capture(self):
        self.info_label.config(text="Kameraya bakın...")
        self.is_running = True
        self.update_frame()

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            self.stop_capture()
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            status_message = "Kameraya Ortalanın"
            current_name = "Bilinmiyor"
            current_conf_percent = 0

            # Önce görüntüyü göster
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]

                if self.mode == 'register':
                    self.progress_label.config(text=f"{self.captured_images}/{self.required_images} yüz verisi")
                    if self.captured_images < self.required_images:
                        user_dir = os.path.join(DATA_DIR, self.user_name)
                        os.makedirs(user_dir, exist_ok=True)
                        img_name = f"{self.user_name}_{time.time()}.png"
                        img_path = os.path.join(user_dir, img_name)
                        cv2.imwrite(img_path, roi_gray)
                        self.captured_images += 1
                        self.progress_label.config(text=f"{self.captured_images}/{self.required_images} yüz verisi")
                        time.sleep(0.3)

                    if self.captured_images >= self.required_images:
                        self.parent_window.event_generate("<<RegistrationComplete>>")
                        self.capture_window.after(2000, self.fade_and_close)

                elif self.mode == 'login' and not self.login_handled:
                    if recognizer is not None:
                        try:
                            id_, conf = recognizer.predict(roi_gray)
                        except Exception:
                            self.result_label.config(text="Hata: Yüz tanıma başarısız.", fg="red")
                            self.login_handled = True
                            self.capture_window.after(2000, self.fade_and_close)
                            return

                        current_conf_percent = round(conf)

                        if conf <= CONFIDENCE_THRESHOLD and id_ in labels:
                            current_name = labels[id_]
                            self.detected_name = current_name
                            self.confidence_score = conf
                            self.result_label.config(text=f"Tanınan: {current_name} (Mesafe: {current_conf_percent})",
                                                     fg="green")
                            self.parent_window.event_generate("<<LoginSuccess>>")
                            self.login_handled = True
                            self.capture_window.after(2000, self.fade_and_close)
                        else:
                            self.result_label.config(
                                text=f"Giriş Başarısız. Tanınan: {current_name}, Mesafe: {current_conf_percent}",
                                fg="red")
                            self.parent_window.event_generate("<<LoginFailed>>")
                            self.login_handled = True
                            self.capture_window.after(2000, self.fade_and_close)
                    else:
                        self.result_label.config(text="Hata: Eğitilmiş model yüklenemedi.", fg="red")
                        self.capture_window.after(2000, self.fade_and_close)

            elif len(faces) > 1:
                self.result_label.config(text="HATA: Birden fazla yüz algılandı!", fg="red")
            else:
                self.result_label.config(text="")

            self.info_label.config(text=status_message)

        except Exception as e:
            self.info_label.config(text=f"HATA: {e}", fg="red")
            self.stop_capture()
            return

        if self.is_running:
            self.capture_window.after(10, self.update_frame)
        else:
            self.cleanup()

    def fade_and_close(self):
        def fade_out(alpha):
            if alpha <= 0:
                if self.mode == 'login' and self.login_handled:
                    if self.detected_name != "Bilinmiyor":
                        # Sadece burada mesaj göster
                        if self.detected_name.lower() == ADMIN_USER:
                            pass  # Admin için mesaj gösterme
                        else:
                            messagebox.showinfo("Başarılı", f"Hoş geldiniz, {self.detected_name}!",
                                                parent=self.capture_window)
                self.stop_capture()
            else:
                self.capture_window.attributes("-alpha", alpha)
                self.capture_window.after(50, lambda: fade_out(alpha - 0.1))

        fade_out(1.0)

    def stop_capture(self):
        if self.is_running:
            self.is_running = False
            if self.mode == 'register' and self.captured_images < self.required_images:
                self.parent_window.event_generate("<<CaptureCancelled>>")
            elif self.mode == 'login' and self.captured_images < self.required_images:
                self.parent_window.event_generate("<<CaptureCancelled>>")

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        try:
            self.capture_window.grab_release()
            self.capture_window.destroy()
        except tk.TclError:
            pass

# --- Ana Uygulama Penceresi ---
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dershane Otomasyonu - Giriş")
        self.root.geometry("500x350")
        self.root.configure(bg="white")
        self.root.resizable(False, False)

        # Pencereyi ekranın ortasına yerleştir (Kod Aynı)
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        self.title_label = tk.Label(root, text="Giriş / Kayıt", font=(FONT_FAMILY, FONT_SIZE + 4, "bold"), bg="white", fg=DARK_VIOLET)
        self.title_label.pack(pady=20)

        self.name_label = tk.Label(root, text="Ad:", font=CUSTOM_FONT, bg="white")
        self.name_label.pack(pady=5)
        self.name_entry = tk.Entry(root, font=CUSTOM_FONT, width=30)
        self.name_entry.pack(pady=5)

        self.surname_label = tk.Label(root, text="Soyad:", font=CUSTOM_FONT, bg="white")
        self.surname_label.pack(pady=5)
        self.surname_entry = tk.Entry(root, font=CUSTOM_FONT, width=30)
        self.surname_entry.pack(pady=5)

        self.action_button = tk.Button(root, text="Giriş Yap / Kayıt Ol", command=self.handle_action, font=CUSTOM_FONT, bg=DARK_VIOLET, fg="white")
        self.action_button.pack(pady=20)

        # Giriş yapan kullanıcıyı saklamak için
        self.current_login_user = None

        # Olay Yöneticileri
        self.root.bind("<<RegistrationComplete>>", self.on_registration_complete)
        self.root.bind("<<LoginSuccess>>", self.on_login_success)
        self.root.bind("<<LoginFailed>>", self.on_login_failed)
        self.root.bind("<<CaptureCancelled>>", self.on_capture_cancelled)

        # Başlangıçta modeli yükle
        load_trained_data()


    def handle_action(self):
        name = self.name_entry.get().strip()
        surname = self.surname_entry.get().strip()

        # Önce Admin kontrolü
        if name.lower() == ADMIN_USER and not surname:
            user_name = ADMIN_USER
            admin_dir = os.path.join(DATA_DIR, ADMIN_USER)
            if not os.path.exists(admin_dir):
                # Admin ilk kez kayıt oluyor
                messagebox.showinfo("Admin Kayıt", f"'{ADMIN_USER}' olarak ilk kayıt işlemi yapılacak. Lütfen {REQUIRED_REGISTER_IMAGES} adet yüz verisi sağlayın.", parent=self.root)
                self.start_capture('register', user_name, REQUIRED_REGISTER_IMAGES)
            else:
                # Admin giriş yapıyor
                self.start_capture('login', user_name, REQUIRED_LOGIN_IMAGES)

        # Normal kullanıcı kontrolü
        else:
            if not name or not surname:
                messagebox.showwarning("Eksik Bilgi", "Lütfen adınızı ve soyadınızı giriniz.", parent=self.root)
                return

            # Kullanıcı adını oluştur (Türkçe karakterleri de koruyabiliriz veya değiştirebiliriz)
            # Şimdilik boşlukları alt çizgi yapalım
            user_name = f"{name}_{surname}".replace(" ", "_")
            user_dir = os.path.join(DATA_DIR, user_name)

            if os.path.exists(user_dir):
                # Kullanıcı var, giriş yapmayı dene
                self.start_capture('login', user_name, REQUIRED_LOGIN_IMAGES)
            else:
                # Kullanıcı yok, kayıt yap
                result = messagebox.askyesno("Kayıt", f"'{name} {surname}' isimli kullanıcı bulunamadı. Kayıt olmak ister misiniz? ({REQUIRED_REGISTER_IMAGES} yüz verisi gereklidir)", parent=self.root)
                if result:
                    self.start_capture('register', user_name, REQUIRED_REGISTER_IMAGES)


    def start_capture(self, mode, user_name, num_images):
        # Ana pencereyi geçici olarak devre dışı bırakmak yerine
        # FaceCaptureApp içindeki grab_set() yeterli olabilir.
        # self.root.attributes('-disabled', True) # Bunu kullanmayalım, grab_set daha iyi

        # Giriş yapan kullanıcıyı sakla
        self.current_login_user = user_name if mode == 'login' else None

        self.capture_app = FaceCaptureApp(self.root, mode, user_name, num_images)


    def on_registration_complete(self, event):
        print("Ana pencere: Kayıt tamamlandı sinyali alındı.")
        if train_model():
             messagebox.showinfo("Model Eğitildi", "Yeni verilerle model başarıyla eğitildi.", parent=self.root)
        else:
             messagebox.showwarning("Model Eğitilemedi", "Model eğitimi sırasında bir sorun oluştu.", parent=self.root)
        # self.root.attributes('-disabled', False) # Devre dışı bırakmadıysak etkinleştirmeye gerek yok
        self.root.focus_set() # Odağı geri al

    def on_login_success(self, event):
        print("Ana pencere: Giriş başarılı sinyali alındı.")
        self.root.focus_set()

        if self.current_login_user == ADMIN_USER:
            print("Admin girişi başarılı. Dashboard açılıyor.")
            self.show_admin_dashboard()
        else:
            # Normal kullanıcı için mesaj artık fade_and_close'da gösteriliyor
            pass

        # Başarılı giriş sonrası kullanıcı adı alanlarını temizle
        self.name_entry.delete(0, tk.END)
        self.surname_entry.delete(0, tk.END)


    def on_login_failed(self, event):
        print("Ana pencere: Giriş başarısız sinyali alındı.")
        # Giriş başarısız olduğunda yapılacaklar (mesaj zaten FaceCaptureApp'te gösterildi)
        # self.root.attributes('-disabled', False) # Etkinleştir
        self.root.focus_set()
        # Başarısız giriş sonrası alanları temizlemeye gerek olmayabilir


    def on_capture_cancelled(self, event):
        print("Ana pencere: Yakalama iptal edildi sinyali alındı.")
        # Kullanıcı yakalama penceresini kapattığında
        # self.root.attributes('-disabled', False) # Etkinleştir
        self.root.focus_set()

    # --- Gerekli importlara ekleme ---
    from tkinter import Canvas, BOTH, YES, NW  # Canvas ve diğer sabitler için

    # ... (Kodun geri kalanı aynı) ...

    # --- Ana Uygulama Sınıfı ---
    class MainApp:
        # ... ( __init__ , handle_action, start_capture metodları aynı) ...
        # ... ( on_registration_complete, on_login_success*, on_login_failed, on_capture_cancelled metodları aynı - *on_login_success içinde show_admin_dashboard çağrısı kalıyor ) ...

        # ... (MainApp sınıfının diğer metodları aynı) ...

        # ... (MainApp sınıfının diğer metodları aynı) ...

        def show_admin_dashboard(self):
            dashboard_window = tk.Toplevel(self.root)
            dashboard_window.title("Admin Paneli - Kayıtlı Kullanıcılar ve Yüz Verileri")
            dashboard_window.geometry("600x600")  # Pencereyi biraz büyütelim
            dashboard_window.configure(bg="white")
            dashboard_window.transient(self.root)
            dashboard_window.grab_set()

            title_label = tk.Label(dashboard_window, text="Kayıtlı Kullanıcılar ve Yüz Örnekleri",
                                   font=(FONT_FAMILY, FONT_SIZE, "bold"), bg="white", fg=DARK_VIOLET)
            title_label.pack(pady=10)

            # --- Kaydırılabilir Çerçeve Oluşturma ---
            main_frame = Frame(dashboard_window, bg="white")
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            my_canvas = tk.Canvas(main_frame, bg="white", highlightthickness=0)
            my_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar = Scrollbar(main_frame, orient="vertical", command=my_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill="y")

            my_canvas.configure(yscrollcommand=scrollbar.set)

            user_display_frame = Frame(my_canvas, bg="white")
            my_canvas.create_window((0, 0), window=user_display_frame, anchor=tk.NW)

            print("--- Admin Dashboard Açılıyor ---")  # DEBUG

            # Scrollbar'ı ayarlamak için iç çerçevenin boyut değişikliklerini dinle
            def on_frame_configure(event):
                # --- DEĞİŞİKLİK BURADA ---
                # Canvas'ın kaydırma bölgesini, içeriği tutan çerçevenin
                # hesaplanan gerekli boyutlarına göre ayarla.
                # bbox("all") yerine winfo_reqwidth/height kullan.
                scroll_width = user_display_frame.winfo_reqwidth()
                scroll_height = user_display_frame.winfo_reqheight()
                my_canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
                # --- DEĞİŞİKLİK SONU ---
                print(
                    f"DEBUG: Frame configured. Scroll region set to: (0, 0, {scroll_width}, {scroll_height}) based on user_display_frame size")  # DEBUG

            user_display_frame.bind("<Configure>", on_frame_configure)

            def on_mouse_wheel(event):
                # Windows/MacOS için delta, Linux için num (4 yukarı, 5 aşağı)
                if hasattr(event, 'delta'):  # Windows/MacOS
                    delta = -1 * int(event.delta / 120)
                elif event.num == 4:  # Linux yukarı
                    delta = -1
                elif event.num == 5:  # Linux aşağı
                    delta = 1
                else:  # Diğer durumlar?
                    delta = 0

                if delta != 0:
                    my_canvas.yview_scroll(delta, "units")
                    # print(f"DEBUG: Mouse wheel scroll: {delta}") # İsteğe bağlı debug

            # Bind events - Farklı platformlar için
            dashboard_window.bind_all("<MouseWheel>", on_mouse_wheel)
            dashboard_window.bind_all("<Button-4>", on_mouse_wheel)
            dashboard_window.bind_all("<Button-5>", on_mouse_wheel)
            # --- Kaydırılabilir Çerçeve Sonu ---

            # --- Kullanıcıları ve Resimleri Listele ---
            THUMBNAIL_SIZE = (100, 100)
            print(f"DEBUG: Kullanıcılar aranıyor: {DATA_DIR}")  # DEBUG
            try:
                registered_users = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
                registered_users.sort()
                print(f"DEBUG: Bulunan kullanıcı dizinleri: {registered_users}")  # DEBUG

                if not registered_users:
                    no_user_label = tk.Label(user_display_frame, text="Hiç kayıtlı kullanıcı bulunamadı.",
                                             font=LISTBOX_FONT, bg="white")
                    no_user_label.pack(pady=10)
                    print("DEBUG: Kayıtlı kullanıcı bulunamadı.")  # DEBUG
                else:
                    for user_dirname in registered_users:
                        print(f"DEBUG: Kullanıcı işleniyor: {user_dirname}")  # DEBUG
                        user_frame = Frame(user_display_frame, bg="white", bd=1, relief="solid")

                        display_name = user_dirname.replace('_', ' ')
                        name_label = tk.Label(user_frame, text=display_name, font=LISTBOX_FONT, bg="white", anchor="w")
                        name_label.pack(side="left", padx=10, pady=5)

                        user_dir_path = os.path.join(DATA_DIR, user_dirname)
                        first_image_path = None
                        try:
                            for filename in os.listdir(user_dir_path):
                                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                                    first_image_path = os.path.join(user_dir_path, filename)
                                    # print(f"DEBUG:   -> Resim bulundu: {filename}") # İsteğe bağlı debug
                                    break
                        except Exception as list_err:
                            print(f"HATA: {user_dir_path} klasörü okunurken hata: {list_err}")

                        img_label = tk.Label(user_frame, bg="white")
                        img_label.pack(side="right", padx=10, pady=5)

                        if first_image_path:
                            try:
                                img = Image.open(first_image_path)
                                img.thumbnail(THUMBNAIL_SIZE)
                                photo = ImageTk.PhotoImage(img)
                                img_label.config(image=photo)
                                img_label.image = photo  # Referansı sakla!
                            except Exception as e:
                                print(f"HATA: {first_image_path} yüklenemedi/işlenemedi - {e}")
                                img_label.config(text="[Resim Hatalı]", font=("Arial", 8), fg="red")
                        else:
                            # print(f"DEBUG:   -> {user_dirname} için resim bulunamadı.") # İsteğe bağlı debug
                            img_label.config(text="[Resim Yok]", font=("Arial", 8))

                        user_frame.pack(fill="x", pady=5, padx=5)
                        # print(f"DEBUG: {user_dirname} çerçevesi eklendi.") # İsteğe bağlı debug

                    # Döngü bittikten sonra çerçeve boyutunu güncellemeye zorla
                    user_display_frame.update_idletasks()
                    print("--- Kullanıcı listeleme tamamlandı ---")  # DEBUG
                    # on_frame_configure fonksiyonunun çağrılması için update sonrasında manuel tetikleme gerekebilir mi?
                    # Genellikle gerekmez, bind yeterli olmalı.

            except FileNotFoundError:
                print(f"HATA: Veri klasörü bulunamadı: {DATA_DIR}")  # DEBUG
                error_label = tk.Label(user_display_frame, text="Hata: Veri klasörü bulunamadı!", font=LISTBOX_FONT,
                                       bg="white", fg="red")
                error_label.pack(pady=10)
            except Exception as e:
                print(f"HATA: Kullanıcılar listelenirken genel hata: {e}")  # DEBUG
                error_label = tk.Label(user_display_frame, text=f"Hata: Kullanıcılar listelenemedi - {e}",
                                       font=LISTBOX_FONT, bg="white", fg="red")
                error_label.pack(pady=10)

            # Kapat Butonu
            close_button = tk.Button(dashboard_window, text="Kapat", command=dashboard_window.destroy, font=CUSTOM_FONT,
                                     bg="grey", fg="white")
            close_button.pack(pady=10, side="bottom")

        # ... (MainApp sınıfının geri kalanı aynı) ...



# --- Uygulamayı Başlat ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()