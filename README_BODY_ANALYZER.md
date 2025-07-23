# 🏋️‍♂️ Vücut Kompozisyon Analizi Sistemi

Bu sistem fotoğraflardan vücut kompozisyonunu analiz eder ve bölge bazlı değerlendirme yapar.

## 🚀 Hızlı Başlangıç

```bash
# Sanal ortamı aktifleştir
source .venv/bin/activate

# Fotoğraf analizi
python smart_body_analyzer.py foto.jpg
```

## 📸 Desteklenen Fotoğraf Türleri

### 🏃 Tam Vücut Fotoğrafları
- **Özellik**: Baş-ayak arası tam görünür
- **Analiz**: MediaPipe ile 33 nokta pose detection
- **Bölgeler**: Omuz, göğüs, karın, kol, bacak, bel, kalça
- **Çıktı**: Detaylı anatomik analiz

### 💪 Üst Vücut Fotoğrafları  
- **Özellik**: Baş-bel arası görünür
- **Analiz**: Görsel bölge segmentasyonu
- **Bölgeler**: Üst/alt göğüs, üst/alt karın, omuzlar, kollar
- **Çıktı**: Torso odaklı kompozisyon analizi

## 🎯 Analiz Çıktıları

### Renk Kodlaması
- 🟢 **Yeşil (85-100)**: Mükemmel kas gelişimi
- 🟡 **Sarı (70-84)**: İyi seviye, küçük iyileştirmeler
- 🟠 **Turuncu (55-69)**: Orta seviye, odaklanma gerekli
- 🔴 **Kırmızı (0-54)**: Gelişim gerekli, yoğun program

### Rapor İçeriği
- **Genel Skor**: Tüm bölgelerin ortalaması
- **Bölge Skorları**: Her bölge için 100 üzerinden puan
- **Kişisel Öneriler**: Bölge bazlı egzersiz tavsiyeleri
- **Ölçümler**: Vücut oranları ve proporsiyon analizi

## 📁 Dosya Yapısı

```
🤖 smart_body_analyzer.py     # Ana akıllı sistem
🏃 simple_body_analyzer.py    # Tam vücut analizi
💪 torso_analyzer.py          # Üst vücut analizi
🧪 test_body_analyzer.py      # Test sistemi
🔬 debug_photo.py             # Debug araçları
```

## 💡 Kullanım İpuçları

### En İyi Sonuçlar İçin:
1. **Aydınlatma**: Doğal ışık veya iyi aydınlatma
2. **Pozisyon**: Kameraya dönük, düz duruş
3. **Arka Plan**: Düz, kontrast oluşturacak renk
4. **Mesafe**: Tam vücut için 2-3 metre uzaklık
5. **Kıyafet**: Vücut hatlarını gösterecek şekilde

### Fotoğraf Kalitesi:
- **Çözünürlük**: Minimum 600x400 piksel
- **Format**: JPG, PNG destekleniyor
- **Boyut**: Maksimum 10MB

## 📊 Örnek Çıktı

```
🎯 Genel Skor: 78/100
🏆 En güçlü: Üst Göğüs (92/100)
⚠️  Gelişim alanı: Alt Karın (63/100)
👍 İyi gelişim gösteriyor, devam edin!
```

## ⚡ Hızlı Komutlar

```bash
# Test fotoğrafı ile deneme
python test_body_analyzer.py

# Debug modu
python debug_photo.py foto.jpg

# Sadece üst vücut analizi
python torso_analyzer.py foto.jpg

# Sadece tam vücut analizi  
python simple_body_analyzer.py foto.jpg
```

## 🔧 Teknik Detaylar

### Gereksinimler:
- Python 3.8+
- OpenCV 4.0+
- MediaPipe 0.10+
- NumPy 1.20+

### Algoritma:
- **Pose Detection**: MediaPipe Pose Landmarks
- **Body Segmentation**: Anatomik bölge tanımlama
- **Composition Analysis**: Orantı ve proporsiyon hesaplama
- **Visual Feedback**: Renkli heatmap oluşturma

## 🎨 Örnek Sonuçlar

Analiz sonuçları şu dosyalar olarak kaydedilir:
- `torso_analysis_foto.jpg` - Üst vücut analizi
- `fullbody_analysis_foto.jpg` - Tam vücut analizi

Her dosyada:
- Orijinal fotoğraf
- Renkli bölge işaretlemeleri
- Detaylı analiz raporu
- Skor ve öneriler

---

**🏆 Sağlıklı yaşam için vücut kompozisyonunuzu takip edin!**