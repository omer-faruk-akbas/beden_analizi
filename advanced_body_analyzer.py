import cv2
import numpy as np
import mediapipe as mp
import sys
import os

class AdvancedBodyAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gelişmiş renk kodlaması
        self.colors = {
            'excellent_muscle': (0, 255, 0),      # Yeşil - Mükemmel kas gelişimi
            'good_muscle': (0, 255, 200),         # Açık yeşil - İyi kas
            'moderate_muscle': (0, 255, 255),     # Sarı - Orta kas gelişimi
            'low_muscle': (0, 200, 255),          # Turuncu - Düşük kas
            'high_fat': (0, 100, 255),            # Kırmızı - Yüksek yağ
            'moderate_fat': (0, 165, 255),        # Turuncu - Orta yağ
            'low_fat': (100, 255, 200),           # Açık yeşil - Düşük yağ
        }
        
        self.regions = {
            'upper_chest': 'Üst Göğüs (Clavicular)',
            'lower_chest': 'Alt Göğüs (Sternal)',
            'upper_abs': 'Üst Karın',
            'lower_abs': 'Alt Karın',
            'obliques': 'Yan Karın (Oblik)',
            'left_deltoid': 'Sol Omuz (Deltoid)',
            'right_deltoid': 'Sağ Omuz (Deltoid)',
            'left_arm': 'Sol Kol',
            'right_arm': 'Sağ Kol',
            'serratus': 'Serratus (Pilot Wings)'
        }
    
    def analyze_muscle_definition(self, image, region_coords):
        """Kas tanımını görsel analiz ile değerlendir"""
        x1, y1, x2, y2 = region_coords
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 50, 'low_muscle'
        
        # Gri tonlamaya çevir
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Kas tanımı için edge detection
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Kontrast analizi (kas çizgilerinin netliği)
        contrast = np.std(gray_roi)
        
        # Histogram analizi (yağ vs kas dokusu)
        hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
        hist_peak = np.argmax(hist)
        
        # Dokusal analiz
        laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        
        # Skorlama algoritması
        muscle_score = 0
        
        # Edge yoğunluği (kas çizgileri)
        if edge_density > 0.15:
            muscle_score += 30
        elif edge_density > 0.1:
            muscle_score += 20
        elif edge_density > 0.05:
            muscle_score += 10
        
        # Kontrast (kas-yağ ayrımı)
        if contrast > 40:
            muscle_score += 25
        elif contrast > 30:
            muscle_score += 15
        elif contrast > 20:
            muscle_score += 5
        
        # Histogram peak (deri tonu vs yağ dokusu)
        if 80 < hist_peak < 150:  # Kas dokusu tonu
            muscle_score += 20
        elif hist_peak > 180:  # Yağ dokusu (daha açık)
            muscle_score -= 10
        
        # Laplacian variance (detay netliği)
        if laplacian_var > 500:
            muscle_score += 25
        elif laplacian_var > 200:
            muscle_score += 15
        
        # Skor normalizasyonu
        muscle_score = max(0, min(100, muscle_score))
        
        # Durum belirleme
        if muscle_score >= 85:
            status = 'excellent_muscle'
        elif muscle_score >= 70:
            status = 'good_muscle' 
        elif muscle_score >= 50:
            status = 'moderate_muscle'
        else:
            status = 'low_muscle'
        
        return muscle_score, status
    
    def analyze_fat_distribution(self, image, region_coords):
        """Yağ dağılımını değerlendir"""
        x1, y1, x2, y2 = region_coords
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0, 'low_fat'
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Yumuşaklık analizi (yağ dokusu daha yumuşak kenarlar)
        blur = cv2.GaussianBlur(gray_roi, (15, 15), 0)
        smoothness = np.mean(np.abs(gray_roi - blur))
        
        # Yağ dokusu genellikle daha homojen
        homogeneity = 1.0 / (1.0 + np.std(gray_roi))
        
        # Parlaklık analizi (yağ dokusu genellikle daha parlak)
        brightness = np.mean(gray_roi)
        
        fat_score = 0
        
        # Yumuşaklık (düşük kontrast = yağ)
        if smoothness < 10:
            fat_score += 30
        elif smoothness < 20:
            fat_score += 20
        elif smoothness < 30:
            fat_score += 10
        
        # Homojenlik
        if homogeneity > 0.05:
            fat_score += 25
        elif homogeneity > 0.03:
            fat_score += 15
        
        # Parlaklık
        if brightness > 160:
            fat_score += 20
        elif brightness > 140:
            fat_score += 10
        
        # Yağ durumu
        if fat_score >= 60:
            fat_status = 'high_fat'
        elif fat_score >= 35:
            fat_status = 'moderate_fat'
        else:
            fat_status = 'low_fat'
        
        return fat_score, fat_status
    
    def get_body_regions(self, landmarks, img_shape):
        """MediaPipe landmarks kullanarak vücut bölgelerini tanımla"""
        h, w = img_shape[:2]
        regions = {}
        
        try:
            # Temel noktaları al
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            
            # Koordinatları piksel değerlerine çevir
            ls = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            rs = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            lh = (int(left_hip.x * w), int(left_hip.y * h))
            rh = (int(right_hip.x * w), int(right_hip.y * h))
            le = (int(left_elbow.x * w), int(left_elbow.y * h))
            re = (int(right_elbow.x * w), int(right_elbow.y * h))
            
            # Üst göğüs (clavicular) - omuz hizası
            chest_top = max(10, ls[1] - 50)
            chest_mid = ls[1] + 40
            regions['upper_chest'] = (ls[0] - 50, chest_top, rs[0] + 50, chest_mid)
            
            # Alt göğüs (sternal) - orta göğüs
            regions['lower_chest'] = (ls[0] - 30, chest_mid, rs[0] + 30, chest_mid + 80)
            
            # Üst karın
            abs_top = chest_mid + 60
            abs_mid = abs_top + 60
            regions['upper_abs'] = (ls[0], abs_top, rs[0], abs_mid)
            
            # Alt karın
            abs_bottom = min(h - 10, abs_mid + 80)
            regions['lower_abs'] = (lh[0] - 20, abs_mid, rh[0] + 20, abs_bottom)
            
            # Yan karın (obliques)
            regions['obliques'] = (ls[0] - 80, abs_top, ls[0] - 10, abs_bottom)
            
            # Omuzlar (deltoid)
            regions['left_deltoid'] = (ls[0] - 60, ls[1] - 40, ls[0] + 20, ls[1] + 40)
            regions['right_deltoid'] = (rs[0] - 20, rs[1] - 40, rs[0] + 60, rs[1] + 40)
            
            # Kollar
            regions['left_arm'] = (le[0] - 40, le[1] - 30, le[0] + 20, le[1] + 60)
            regions['right_arm'] = (re[0] - 20, re[1] - 30, re[0] + 40, re[1] + 60)
            
            # Serratus (pilot wings) - yan göğüs
            regions['serratus'] = (ls[0] - 40, ls[1] + 20, ls[0], ls[1] + 100)
            
        except Exception as e:
            print(f"Bölge tanımlama hatası: {e}")
            # Varsayılan bölgeler
            regions = self.get_default_regions(img_shape)
            
        return regions
    
    def get_default_regions(self, img_shape):
        """MediaPipe başarısız olursa varsayılan bölgeler"""
        h, w = img_shape[:2]
        
        return {
            'upper_chest': (int(w*0.25), int(h*0.15), int(w*0.75), int(h*0.35)),
            'lower_chest': (int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.5)),
            'upper_abs': (int(w*0.35), int(h*0.45), int(w*0.65), int(h*0.6)),
            'lower_abs': (int(w*0.35), int(h*0.55), int(w*0.65), int(h*0.75)),
            'obliques': (int(w*0.15), int(h*0.45), int(w*0.35), int(h*0.7)),
            'left_deltoid': (int(w*0.1), int(h*0.1), int(w*0.35), int(h*0.4)),
            'right_deltoid': (int(w*0.65), int(h*0.1), int(w*0.9), int(h*0.4)),
            'left_arm': (int(w*0.0), int(h*0.25), int(w*0.25), int(h*0.6)),
            'right_arm': (int(w*0.75), int(h*0.25), int(w*1.0), int(h*0.6)),
            'serratus': (int(w*0.2), int(h*0.3), int(w*0.4), int(h*0.55))
        }
    
    def generate_recommendations(self, analysis_results):
        """Analiz sonuçlarına göre öneriler üret"""
        recommendations = []
        
        # Genel durum değerlendirmesi
        muscle_scores = [r['muscle_score'] for r in analysis_results.values()]
        fat_scores = [r['fat_score'] for r in analysis_results.values()]
        
        avg_muscle = np.mean(muscle_scores)
        avg_fat = np.mean(fat_scores)
        
        if avg_muscle >= 80:
            recommendations.append("🏆 Mükemmel kas gelişimi!")
        elif avg_muscle >= 60:
            recommendations.append("💪 İyi kas gelişimi, devam edin")
        else:
            recommendations.append("🔥 Kas geliştirme odaklı program gerekli")
        
        if avg_fat <= 30:
            recommendations.append("✨ Düşük yağ oranı - ideal")
        elif avg_fat <= 50:
            recommendations.append("⚡ Orta yağ oranı - cutting yapılabilir")
        else:
            recommendations.append("🎯 Yağ yakma programı önerilir")
        
        # Bölge özel öneriler
        weak_regions = [region for region, data in analysis_results.items() 
                       if data['muscle_score'] < 60]
        
        if weak_regions:
            recommendations.append(f"🎯 Odaklanılacak bölgeler: {', '.join([self.regions[r] for r in weak_regions[:3]])}")
        
        return recommendations
    
    def draw_analysis(self, image, regions, analysis_results):
        """Analiz sonuçlarını görsel olarak çiz"""
        overlay = image.copy()
        
        for region_name, coords in regions.items():
            if region_name in analysis_results:
                x1, y1, x2, y2 = coords
                result = analysis_results[region_name]
                
                # Kas durumuna göre renk
                muscle_color = self.colors[result['muscle_status']]
                
                # Yağ durumuna göre transparanlık
                alpha = 0.3 if result['fat_score'] > 50 else 0.4
                
                # Bölgeyi renklendir
                cv2.rectangle(overlay, (x1, y1), (x2, y2), muscle_color, -1)
                
                # Etiket
                label = f"{self.regions[region_name][:10]}: M{result['muscle_score']}/F{result['fat_score']}"
                cv2.putText(overlay, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Şeffaf birleştirme
        result = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        return result
    
    def draw_detailed_report(self, image, analysis_results, recommendations):
        """Detaylı rapor çiz"""
        h, w = image.shape[:2]
        report_w = 350
        
        # Rapor kutusu
        cv2.rectangle(image, (w - report_w - 10, 10), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(image, (w - report_w - 10, 10), (w - 10, h - 10), (255, 255, 255), 2)
        
        # Başlık
        cv2.putText(image, "GELISMIS VUCUT ANALIZI", (w - report_w + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y = 60
        
        # Genel skorlar
        muscle_scores = [r['muscle_score'] for r in analysis_results.values()]
        fat_scores = [r['fat_score'] for r in analysis_results.values()]
        
        avg_muscle = np.mean(muscle_scores)
        avg_fat = np.mean(fat_scores)
        
        cv2.putText(image, f"Ortalama Kas Skoru: {avg_muscle:.0f}/100", 
                   (w - report_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y += 20
        
        cv2.putText(image, f"Ortalama Yag Skoru: {avg_fat:.0f}/100", 
                   (w - report_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        y += 30
        
        # Bölge detayları
        cv2.putText(image, "BOLGE DETAYLARI:", (w - report_w + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y += 20
        
        for region_name, result in analysis_results.items():
            if y > h - 60:
                break
                
            region_display = self.regions[region_name][:15]
            muscle_score = result['muscle_score']
            fat_score = result['fat_score']
            
            # Bölge adı
            cv2.putText(image, region_display, (w - report_w + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            y += 15
            
            # Skorlar
            cv2.putText(image, f"  Kas: {muscle_score}/100", (w - report_w + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 100), 1)
            cv2.putText(image, f"Yag: {fat_score}/100", (w - report_w + 150, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1)
            y += 20
        
        # Öneriler
        y += 10
        cv2.putText(image, "ONERILER:", (w - report_w + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y += 20
        
        for rec in recommendations[:4]:  # Maksimum 4 öneri
            if y > h - 20:
                break
            cv2.putText(image, rec[:35], (w - report_w + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 100), 1)
            y += 15
        
        return image
    
    def analyze_image(self, image_path):
        """Ana analiz fonksiyonu"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Fotoğraf yüklenemedi: {image_path}")
            return None
        
        print(f"🔬 Gelişmiş analiz başlatılıyor: {image_path}")
        print("-" * 60)
        
        # MediaPipe pose detection
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.3
        ) as pose:
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            # Bölgeleri tanımla
            if results.pose_landmarks:
                print("✅ Pose landmarks tespit edildi")
                regions = self.get_body_regions(results.pose_landmarks.landmark, image.shape)
                
                # İskelet çiz
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                )
            else:
                print("⚠️  Pose tespit edilemedi, varsayılan bölgeler kullanılıyor")
                regions = self.get_default_regions(image.shape)
            
            # Her bölgeyi analiz et
            analysis_results = {}
            
            for region_name, coords in regions.items():
                muscle_score, muscle_status = self.analyze_muscle_definition(image, coords)
                fat_score, fat_status = self.analyze_fat_distribution(image, coords)
                
                analysis_results[region_name] = {
                    'muscle_score': muscle_score,
                    'muscle_status': muscle_status,
                    'fat_score': fat_score,
                    'fat_status': fat_status
                }
            
            # Öneriler üret
            recommendations = self.generate_recommendations(analysis_results)
            
            # Görsel analiz çiz
            image = self.draw_analysis(image, regions, analysis_results)
            
            # Detaylı rapor ekle
            image = self.draw_detailed_report(image, analysis_results, recommendations)
            
            return image, analysis_results, recommendations

def main():
    if len(sys.argv) != 2:
        print("🔬 GELİŞMİŞ VÜCUT KOMPOZİSYON ANALİZİ")
        print("=" * 50)
        print("Kullanım:")
        print(f"  python {sys.argv[0]} <foto_yolu>")
        print("\nÖzellikler:")
        print("  🧠 Yapay zeka tabanlı kas-yağ analizi")
        print("  🎯 10 farklı vücut bölgesi analizi")
        print("  📊 Detaylı kompozisyon raporu")
        print("  💡 Kişiselleştirilmiş öneriler")
        return
    
    analyzer = AdvancedBodyAnalyzer()
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    result = analyzer.analyze_image(image_path)
    
    if result is not None:
        image, analysis_results, recommendations = result
        
        # Sonucu kaydet
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"advanced_analysis_{name}{ext}"
        cv2.imwrite(output_path, image)
        
        print(f"✅ BAŞARILI! Sonuç: {output_path}")
        
        # Konsol raporu
        muscle_scores = [r['muscle_score'] for r in analysis_results.values()]
        fat_scores = [r['fat_score'] for r in analysis_results.values()]
        
        print(f"\n🎯 GENEL DEĞERLENDIRME:")
        print(f"   Ortalama Kas Skoru: {np.mean(muscle_scores):.0f}/100")
        print(f"   Ortalama Yağ Skoru: {np.mean(fat_scores):.0f}/100")
        
        print(f"\n💡 ÖNERİLER:")
        for rec in recommendations:
            print(f"   {rec}")
        
        # En iyi ve en zayıf bölgeler
        best_muscle = max(analysis_results.items(), key=lambda x: x[1]['muscle_score'])
        worst_muscle = min(analysis_results.items(), key=lambda x: x[1]['muscle_score'])
        
        print(f"\n🏆 En güçlü bölge: {analyzer.regions[best_muscle[0]]} ({best_muscle[1]['muscle_score']}/100)")
        print(f"⚠️  Gelişim alanı: {analyzer.regions[worst_muscle[0]]} ({worst_muscle[1]['muscle_score']}/100)")

if __name__ == "__main__":
    main()