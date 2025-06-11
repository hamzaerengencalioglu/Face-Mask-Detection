# 😷 Face Mask Detection

Bu proje, bir kişinin yüzünde **maske takılı olup olmadığını** tespit eden, gerçek zamanlı çalışan bir bilgisayarla görü sistemidir. Sistem, **OpenCV** tabanlı yüz algılama modeli ile **Keras + TensorFlow** kullanılarak eğitilen **MobileNetV2** temelli bir derin öğrenme modelini birleştirir. Ayrıca, proje bir masaüstü uygulaması olarak kullanılabilmesi için **PyQt5** ile tasarlanmış GUI arayüzüne sahiptir.

---

## 📌 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Eğitimi (Kendi Veri Kümenizle)](#model-eğitimi-kendi-veri-kümenizle)
- [Klasör Yapısı](#klasör-yapısı)
- [Notlar](#notlar)

---

## 📖 Proje Hakkında

Face Mask Detection, kameradan alınan görüntü üzerinden veya yüklenen bir görsel üzerinden kişilerin maske takıp takmadığını tespit eden bir görüntü işleme uygulamasıdır. COVID-19 gibi salgın süreçlerinde kamusal alanlarda maske denetimini otomatikleştirmek için kullanılabilir.

Model, **MobileNetV2** kullanılarak transfer learning yöntemiyle eğitilmiştir. Giriş olarak yüz görüntüsü alır ve "Maskeli" ya da "Maskesiz" sınıfına ait olup olmadığını tahmin eder.

---

## 🧰 Kullanılan Teknolojiler

- **Python 3.x**
- **TensorFlow / Keras** – Derin öğrenme modeli (MobileNetV2)
- **OpenCV** – Yüz algılama (DNN modülü)
- **PyQt5** – Masaüstü GUI arayüzü
- **Scikit-learn** – Veri ön işleme ve değerlendirme
- **Imutils** – Görüntü işleme yardımcıları
- **Matplotlib** – Eğitim sonrası görselleştirme

---

## ⚙️ Kurulum

```bash
# 1. Repoyu klonlayın
git clone https://github.com/hamzaerengencalioglu/Face-Mask-Detection.git
cd Face-Mask-Detection

# 2. Gereksinimleri kurun
pip install -r requirements.txt
```

---

## 🚀 Kullanım

### Arayüz (GUI) Üzerinden Gerçek Zamanlı Test

```bash
python gui.py
```

> Kamera açılır ve her yüz için "MASK" veya "NO MASK" tahmini yapılır. Arayüz, gerçek zamanlı görüntü üzerinde sonuçları gösterir.

---

## 🧠 Model Eğitimi (Kendi Veri Kümenizle)

Kendi veri kümenizle eğitmek için, veri klasörünüz şu şekilde yapılandırılmalıdır:

```
dataset/
├── with_mask/
│   ├── img001.jpg
│   ├── ...
├── without_mask/
│   ├── img201.jpg
│   ├── ...
```

### Eğitim Komutu

```bash
python train_mask_detector.py --dataset dataset/
```

### Eğitim Süreci Özeti

- **Model**: MobileNetV2 (imagenet ağırlıkları, `include_top=False`)
- **Input boyutu**: (224, 224, 3)
- **Augmentasyon**: Dönme, zoom, kaydırma, çevirme
- **Optimizasyon**: Adam optimizer (learning_rate = 1e-4)
- **Epoch**: 20
- **Batch size**: 32
- **Kayıtlı Çıktılar**:
  - `mask_detector_model.h5`: Eğitim sonrası model
  - `plot.png`: Accuracy / loss grafik çıktısı
  - Konsolda sınıflandırma raporu

---

## 📁 Klasör Yapısı

```bash
Face-Mask-Detection/
│
├── face_detector/                  # Yüz tespiti için OpenCV DNN modeli
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── mask_detector_model.h5         # Eğitilmiş model (MobileNetV2)
├── gui.py                         # PyQt5 GUI arayüz
├── train_mask_detector.py         # Model eğitimi için Python betiği
├── users/                         # Kullanıcı verisi için ayrılmış klasör
├── plot.png                       # Eğitim sonucu doğruluk/kayıp grafiği
├── README.md                      # Açıklama dosyası
└── requirements.txt               # Bağımlılık listesi
```

---

## 📝 Notlar

- Model `mask_detector_model.h5` olarak kaydedilir, GUI arayüzü bu dosyayı yükleyerek çalışır.
- `face_detector/` klasöründe OpenCV’nin Caffe modeli yer alır ve yüzlerin koordinatlarını belirler.
- Eğitim kodu tamamen özelleştirilebilir (örneğin: epoch sayısı, dropout oranı, model başlığı vb.).
- Eğitim için kendi veri kümeniz kullanılabilir ve esnek augmentasyon desteklenmektedir.

---

**Hazırlayan:** [hamzaerengencalioglu](https://github.com/hamzaerengencalioglu)
