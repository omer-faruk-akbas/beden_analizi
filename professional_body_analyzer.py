import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

@dataclass
class BodyRegionAnalysis:
    """Vücut bölgesi analiz sonucu"""
    muscle_mass: float  # Kas kitlesi (0-100)
    fat_percentage: float  # Yağ oranı (0-100)
    definition: float  # Tanım netliği (0-100)
    overall_score: float  # Genel skor (0-100)
    status_color: Tuple[int, int, int]  # BGR renk kodu
    recommendations: List[str]

class ProfessionalBodyAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Profesyonel renk paleti
        self.colors = {
            'excellent': (0, 255, 0),      # Yeşil - Mükemmel (85-100)
            'very_good': (0, 200, 100),    # Açık yeşil - Çok iyi (75-84)
            'good': (0, 255, 255),         # Sarı - İyi (65-74)
            'average': (0, 165, 255),      # Turuncu - Orta (50-64)
            'poor': (0, 100, 255),         # Açık kırmızı - Zayıf (35-49)
            'very_poor': (0, 0, 255),      # Kırmızı - Çok zayıf (0-34)
            'background': (50, 50, 50),    # Koyu gri - arka plan
            'text': (255, 255, 255),       # Beyaz - metin
            'border': (200, 200, 200)      # Açık gri - kenarlık
        }
        
        # Anatomik bölge tanımları
        self.anatomical_regions = {
            'upper_pectoralis': 'Üst Göğüs (Klavikula)',
            'lower_pectoralis': 'Alt Göğüs (Sternal)',
            'upper_rectus': 'Üst Karın (2-4 Pak)',
            'lower_rectus': 'Alt Karın (6-8 Pak)',
            'external_obliques': 'Dış Oblik (V-Line)',
            'serratus_anterior': 'Serratus (Pilot Wings)',
            'anterior_deltoid': 'Ön Deltoid',
            'lateral_deltoid': 'Yan Deltoid',
            'biceps_brachii': 'Biceps',
            'triceps_brachii': 'Triceps'
        }
    
    def advanced_muscle_analysis(self, roi: np.ndarray, region_name: str) -> Tuple[float, float, float]:
        """Gelişmiş kas analizi"""
        if roi.size == 0:
            return 0.0, 0.0, 0.0
        
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 1. KONTRAST ANALİZİ (Kas çizgilerinin belirginliği)
        contrast_score = self._analyze_contrast(gray)
        
        # 2. EDGE DENSİTY (Kas liflerinin netliği)
        edge_score = self._analyze_muscle_definition(gray)
        
        # 3. TEXTURE ANALİZİ (Kas dokusunun kalitesi)
        texture_score = self._analyze_muscle_texture(gray)
        
        # 4. GRADIENT ANALİZİ (Kas boyutunun derinlik hissi)
        gradient_score = self._analyze_muscle_volume(gray)
        
        # 5. BÖLGEYE ÖZEL AĞIRLIK KATSAYILARI
        weights = self._get_region_weights(region_name)
        
        # SKORLAMA
        muscle_mass = (
            contrast_score * weights['contrast'] +
            edge_score * weights['definition'] +
            texture_score * weights['texture'] +
            gradient_score * weights['volume']
        )
        
        definition = (edge_score * 0.4 + contrast_score * 0.6)
        overall = (muscle_mass * 0.7 + definition * 0.3)
        
        return min(100, max(0, muscle_mass)), min(100, max(0, definition)), min(100, max(0, overall))
    
    def advanced_fat_analysis(self, roi: np.ndarray, region_name: str) -> float:
        """Gelişmiş yağ analizi"""
        if roi.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 1. SMOOTHNESS İNDEKSİ (Yağ dokusunun yumuşaklığı)
        smoothness = self._analyze_fat_smoothness(gray)
        
        # 2. BRIGHTNESS ANALİZİ (Yağ dokusunun parlaklığı)
        brightness = self._analyze_fat_brightness(gray)
        
        # 3. HOMOGENEITY (Yağ dokusunun homojenliği)
        homogeneity = self._analyze_fat_homogeneity(gray)
        
        # 4. THICKNESS İNDİKATÖRÜ (Yağ tabakası kalınlığı)
        thickness = self._analyze_fat_thickness(gray)
        
        # Bölgeye özel yağ analizi
        fat_weights = self._get_fat_weights(region_name)
        
        fat_percentage = (
            smoothness * fat_weights['smoothness'] +
            brightness * fat_weights['brightness'] +
            homogeneity * fat_weights['homogeneity'] +
            thickness * fat_weights['thickness']
        )
        
        return min(100, max(0, fat_percentage))
    
    def _analyze_contrast(self, gray: np.ndarray) -> float:
        """Kontrast analizi - kas çizgilerinin belirginliği"""
        # Standart sapma ile kontrast
        std_contrast = np.std(gray)
        
        # Michelson kontrast
        max_val, min_val = np.max(gray), np.min(gray)
        michelson = (max_val - min_val) / (max_val + min_val + 1e-6)
        
        # RMS kontrast
        mean_val = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_val) ** 2))
        
        # Skorlama
        score = (std_contrast / 255.0 * 40 + 
                michelson * 35 + 
                rms_contrast / 255.0 * 25)
        
        return score
    
    def _analyze_muscle_definition(self, gray: np.ndarray) -> float:
        """Kas tanımı analizi"""
        # Canny edge detection
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_strength = np.mean(magnitude)
        
        # Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Skorlama
        score = (edge_density * 100 * 0.4 + 
                edge_strength / 255.0 * 100 * 0.35 + 
                min(laplacian_var / 1500.0 * 100, 100) * 0.25)
        
        return score
    
    def _analyze_muscle_texture(self, gray: np.ndarray) -> float:
        """Kas dokusu analizi"""
        # GLCM (Gray Level Co-occurrence Matrix) simülasyonu
        # Basitleştirilmiş texture features
        
        # Local Binary Pattern benzeri analiz
        texture_variance = 0
        h, w = gray.shape
        
        if h > 4 and w > 4:
            for i in range(2, h-2):
                for j in range(2, w-2):
                    center = gray[i, j]
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j-1], gray[i, j+1],
                        gray[i+1, j-1], gray[i+1, j], gray[i+1, j+1]
                    ]
                    texture_variance += np.var(neighbors)
            
            texture_variance /= ((h-4) * (w-4))
        
        # Gabor filter response (kas lifi yönelimi)
        kernel = cv2.getGaborKernel((15, 15), 3, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_strength = np.mean(gabor_response)
        
        score = min(texture_variance / 500.0 * 60 + gabor_strength / 255.0 * 40, 100)
        return score
    
    def _analyze_muscle_volume(self, gray: np.ndarray) -> float:
        """Kas hacmi/derinlik analizi"""
        # Gradient magnitude (3D form hissi)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Shadow/highlight analizi (kas hacmi göstergesi)
        mean_intensity = np.mean(gray)
        highlights = np.sum(gray > mean_intensity + 30)
        shadows = np.sum(gray < mean_intensity - 30)
        volume_indicator = (highlights + shadows) / gray.size
        
        # Histogram range (dinamik aralık)
        hist_range = np.max(gray) - np.min(gray)
        
        score = (np.mean(gradient_magnitude) / 255.0 * 40 +
                volume_indicator * 100 * 0.35 +
                hist_range / 255.0 * 25)
        
        return score
    
    def _analyze_fat_smoothness(self, gray: np.ndarray) -> float:
        """Yağ yumuşaklığı analizi"""
        # Gaussian blur ile smoothness
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        smoothness = 100 - np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0 * 100
        
        # High-frequency noise (kas dokusu olmaması)
        high_freq = gray.astype(float) - blurred.astype(float)
        noise_level = np.std(high_freq)
        
        # Düşük noise = yüksek yağ
        fat_score = smoothness * 0.7 + (100 - min(noise_level / 20.0 * 100, 100)) * 0.3
        return fat_score
    
    def _analyze_fat_brightness(self, gray: np.ndarray) -> float:
        """Yağ parlaklığı analizi"""
        mean_brightness = np.mean(gray)
        
        # Yağ dokusu genellikle daha parlak
        if mean_brightness > 160:
            return min((mean_brightness - 160) / 95.0 * 100, 100)
        elif mean_brightness > 120:
            return (mean_brightness - 120) / 40.0 * 60
        else:
            return 0
    
    def _analyze_fat_homogeneity(self, gray: np.ndarray) -> float:
        """Yağ homojenliği analizi"""
        # Standart sapma (düşük = homojen = yağ)
        homogeneity = 100 - min(np.std(gray) / 40.0 * 100, 100)
        
        # Entropy (düşük = homojen)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        entropy_score = 100 - min(entropy / 8.0 * 100, 100)
        
        return homogeneity * 0.6 + entropy_score * 0.4
    
    def _analyze_fat_thickness(self, gray: np.ndarray) -> float:
        """Yağ tabakası kalınlığı analizi"""
        # Gradient magnitude (düşük = kalın yağ tabakası)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Düşük gradient = kalın yağ
        thickness_score = 100 - min(np.mean(gradient_magnitude) / 100.0 * 100, 100)
        
        return thickness_score
    
    def _get_region_weights(self, region_name: str) -> Dict[str, float]:
        """Bölgeye özel kas analiz ağırlıkları"""
        weights = {
            'upper_pectoralis': {'contrast': 0.35, 'definition': 0.25, 'texture': 0.2, 'volume': 0.2},
            'lower_pectoralis': {'contrast': 0.3, 'definition': 0.3, 'texture': 0.2, 'volume': 0.2},
            'upper_rectus': {'contrast': 0.4, 'definition': 0.35, 'texture': 0.15, 'volume': 0.1},
            'lower_rectus': {'contrast': 0.4, 'definition': 0.35, 'texture': 0.15, 'volume': 0.1},
            'external_obliques': {'contrast': 0.25, 'definition': 0.4, 'texture': 0.2, 'volume': 0.15},
            'serratus_anterior': {'contrast': 0.2, 'definition': 0.5, 'texture': 0.2, 'volume': 0.1},
            'anterior_deltoid': {'contrast': 0.3, 'definition': 0.25, 'texture': 0.25, 'volume': 0.2},
            'lateral_deltoid': {'contrast': 0.25, 'definition': 0.25, 'texture': 0.25, 'volume': 0.25},
            'biceps_brachii': {'contrast': 0.3, 'definition': 0.3, 'texture': 0.2, 'volume': 0.2},
            'triceps_brachii': {'contrast': 0.3, 'definition': 0.3, 'texture': 0.2, 'volume': 0.2}
        }
        return weights.get(region_name, {'contrast': 0.3, 'definition': 0.3, 'texture': 0.2, 'volume': 0.2})
    
    def _get_fat_weights(self, region_name: str) -> Dict[str, float]:
        """Bölgeye özel yağ analiz ağırlıkları"""
        weights = {
            'upper_pectoralis': {'smoothness': 0.3, 'brightness': 0.25, 'homogeneity': 0.25, 'thickness': 0.2},
            'lower_pectoralis': {'smoothness': 0.35, 'brightness': 0.25, 'homogeneity': 0.2, 'thickness': 0.2},
            'upper_rectus': {'smoothness': 0.2, 'brightness': 0.3, 'homogeneity': 0.3, 'thickness': 0.2},
            'lower_rectus': {'smoothness': 0.25, 'brightness': 0.3, 'homogeneity': 0.25, 'thickness': 0.2},
            'external_obliques': {'smoothness': 0.4, 'brightness': 0.2, 'homogeneity': 0.2, 'thickness': 0.2}
        }
        return weights.get(region_name, {'smoothness': 0.3, 'brightness': 0.25, 'homogeneity': 0.25, 'thickness': 0.2})
    
    def get_precise_body_regions(self, landmarks, img_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
        """Hassas anatomik bölge koordinatları"""
        h, w = img_shape[:2]
        regions = {}
        
        try:
            # MediaPipe landmark noktaları
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Koordinat dönüşümü
            ls = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            rs = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            lh = (int(left_hip.x * w), int(left_hip.y * h))
            rh = (int(right_hip.x * w), int(right_hip.y * h))
            le = (int(left_elbow.x * w), int(left_elbow.y * h))
            re = (int(right_elbow.x * w), int(right_elbow.y * h))
            lw = (int(left_wrist.x * w), int(left_wrist.y * h))
            rw = (int(right_wrist.x * w), int(right_wrist.y * h))
            
            # Torso merkezi ve genişlik
            torso_center_x = (ls[0] + rs[0]) // 2
            torso_width = abs(rs[0] - ls[0])
            
            # ÜST GÖĞÜS (Clavicular Pectoralis)
            chest_top = max(10, ls[1] - 40)
            chest_upper_bottom = ls[1] + 30
            regions['upper_pectoralis'] = (
                torso_center_x - torso_width//3,
                chest_top,
                torso_center_x + torso_width//3,
                chest_upper_bottom
            )
            
            # ALT GÖĞÜS (Sternal Pectoralis)
            chest_lower_top = chest_upper_bottom - 10
            chest_lower_bottom = chest_lower_top + 70
            regions['lower_pectoralis'] = (
                torso_center_x - int(torso_width * 0.4),
                chest_lower_top,
                torso_center_x + int(torso_width * 0.4),
                chest_lower_bottom
            )
            
            # ÜST KARIN (Upper Rectus Abdominis)
            abs_top = chest_lower_bottom - 20
            abs_mid = abs_top + 60
            regions['upper_rectus'] = (
                torso_center_x - torso_width//4,
                abs_top,
                torso_center_x + torso_width//4,
                abs_mid
            )
            
            # ALT KARIN (Lower Rectus Abdominis)
            abs_lower_top = abs_mid - 10
            abs_lower_bottom = min(h - 50, (lh[1] + rh[1]) // 2)
            regions['lower_rectus'] = (
                torso_center_x - torso_width//4,
                abs_lower_top,
                torso_center_x + torso_width//4,
                abs_lower_bottom
            )
            
            # DIŞ OBLİK (External Obliques) - Love Handles
            oblique_top = abs_top
            oblique_bottom = abs_lower_bottom
            regions['external_obliques'] = (
                ls[0] - 60,
                oblique_top,
                ls[0] - 10,
                oblique_bottom
            )
            
            # SERRATUS ANTERİOR (Pilot Wings)
            serratus_top = ls[1] + 20
            serratus_bottom = serratus_top + 80
            regions['serratus_anterior'] = (
                ls[0] - 30,
                serratus_top,
                ls[0] + 10,
                serratus_bottom
            )
            
            # ÖN DELTOİD (Anterior Deltoid)
            deltoid_size = int(torso_width * 0.15)
            regions['anterior_deltoid'] = (
                ls[0] - deltoid_size,
                ls[1] - 30,
                ls[0] + deltoid_size,
                ls[1] + 50
            )
            
            # YAN DELTOİD (Lateral Deltoid)
            regions['lateral_deltoid'] = (
                ls[0] - deltoid_size - 20,
                ls[1] - 20,
                ls[0] - 20,
                ls[1] + 60
            )
            
            # BİCEPS
            bicep_width = abs(le[0] - ls[0]) // 3
            regions['biceps_brachii'] = (
                le[0] - bicep_width,
                ls[1] + 20,
                le[0] + bicep_width,
                le[1] + 20
            )
            
            # TRİCEPS (arka kol tahmini)
            regions['triceps_brachii'] = (
                le[0] - bicep_width + 20,
                ls[1] + 20,
                le[0] + bicep_width + 20,
                le[1] + 20
            )
            
        except Exception as e:
            print(f"⚠️  Landmark hatası: {e}")
            regions = self._get_default_precise_regions(img_shape)
        
        # Sınır kontrolü
        for region_name, (x1, y1, x2, y2) in regions.items():
            regions[region_name] = (
                max(0, min(x1, w-1)),
                max(0, min(y1, h-1)),
                max(0, min(x2, w-1)),
                max(0, min(y2, h-1))
            )
        
        return regions
    
    def _get_default_precise_regions(self, img_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
        """Varsayılan hassas bölgeler"""
        h, w = img_shape[:2]
        return {
            'upper_pectoralis': (int(w*0.3), int(h*0.12), int(w*0.7), int(h*0.28)),
            'lower_pectoralis': (int(w*0.28), int(h*0.25), int(w*0.72), int(h*0.45)),
            'upper_rectus': (int(w*0.4), int(h*0.4), int(w*0.6), int(h*0.55)),
            'lower_rectus': (int(w*0.4), int(h*0.5), int(w*0.6), int(h*0.7)),
            'external_obliques': (int(w*0.15), int(h*0.4), int(w*0.35), int(h*0.65)),
            'serratus_anterior': (int(w*0.2), int(h*0.3), int(w*0.35), int(h*0.5)),
            'anterior_deltoid': (int(w*0.12), int(h*0.08), int(w*0.28), int(h*0.25)),
            'lateral_deltoid': (int(w*0.05), int(h*0.1), int(w*0.2), int(h*0.3)),
            'biceps_brachii': (int(w*0.08), int(h*0.25), int(w*0.22), int(h*0.45)),
            'triceps_brachii': (int(w*0.12), int(h*0.25), int(w*0.26), int(h*0.45))
        }
    
    def determine_status_color(self, overall_score: float) -> Tuple[int, int, int]:
        """Skor bazlı renk belirleme"""
        if overall_score >= 85:
            return self.colors['excellent']      # Yeşil
        elif overall_score >= 75:
            return self.colors['very_good']      # Açık yeşil
        elif overall_score >= 65:
            return self.colors['good']           # Sarı
        elif overall_score >= 50:
            return self.colors['average']        # Turuncu
        elif overall_score >= 35:
            return self.colors['poor']           # Açık kırmızı
        else:
            return self.colors['very_poor']      # Kırmızı
    
    def generate_professional_recommendations(self, analysis: Dict[str, BodyRegionAnalysis]) -> List[str]:
        """Profesyonel öneriler üret"""
        recommendations = []
        
        # Genel değerlendirme
        muscle_scores = [region.muscle_mass for region in analysis.values()]
        fat_scores = [region.fat_percentage for region in analysis.values()]
        overall_scores = [region.overall_score for region in analysis.values()]
        
        avg_muscle = np.mean(muscle_scores)
        avg_fat = np.mean(fat_scores)
        avg_overall = np.mean(overall_scores)
        
        # Genel durum
        if avg_overall >= 80:
            recommendations.append("🏆 Elite seviye vücut kompozisyonu")
        elif avg_overall >= 70:
            recommendations.append("💪 İleri seviye fizik gelişimi")
        elif avg_overall >= 60:
            recommendations.append("📈 İyi gelişim trendi, optimizasyon gerekli")
        elif avg_overall >= 45:
            recommendations.append("🎯 Orta seviye, odaklanılacak alanlar mevcut")
        else:
            recommendations.append("🔥 Başlangıç seviyesi, kapsamlı program gerekli")
        
        # Kas gelişimi önerileri
        if avg_muscle < 50:
            recommendations.append("💥 Hipertrofi odaklı antrenman programı")
            recommendations.append("🍖 Protein alımını artır (2g/kg vücut ağırlığı)")
        elif avg_muscle < 70:
            recommendations.append("⚡ Progresif overload ilkesini uygula")
            
        # Yağ oranı önerileri
        if avg_fat > 60:
            recommendations.append("🔥 Cutting fazı: Kalori açığı + kardiyovasküler")
            recommendations.append("🥗 Beslenme planını gözden geçir")
        elif avg_fat > 40:
            recommendations.append("⚖️ Body recomposition: Kas koruma + yağ yakma")
        
        # Zayıf bölgeler
        weak_regions = [(name, region) for name, region in analysis.items() 
                       if region.overall_score < 55]
        
        if weak_regions:
            weak_names = [self.anatomical_regions[name] for name, _ in weak_regions[:3]]
            recommendations.append(f"🎯 Öncelik bölgeleri: {', '.join(weak_names)}")
        
        # Güçlü bölgeler
        strong_regions = [(name, region) for name, region in analysis.items() 
                         if region.overall_score > 75]
        
        if strong_regions:
            strong_names = [self.anatomical_regions[name] for name, _ in strong_regions[:2]]
            recommendations.append(f"✨ Güçlü bölgeler: {', '.join(strong_names)}")
        
        return recommendations[:8]  # Maksimum 8 öneri
    
    def draw_professional_visualization(self, image: np.ndarray, regions: Dict[str, Tuple[int, int, int, int]], 
                                      analysis: Dict[str, BodyRegionAnalysis]) -> np.ndarray:
        """Profesyonel görselleştirme"""
        overlay = image.copy()
        result = image.copy()
        
        # Bölgeleri renklendir
        for region_name, (x1, y1, x2, y2) in regions.items():
            if region_name in analysis:
                region_analysis = analysis[region_name]
                
                # Bölgeyi renklendir
                cv2.rectangle(overlay, (x1, y1), (x2, y2), region_analysis.status_color, -1)
                
                # Kenarlık çiz
                cv2.rectangle(result, (x1, y1), (x2, y2), self.colors['border'], 2)
                
                # Bölge etiketi
                region_display = self.anatomical_regions[region_name][:12]
                label_y = y1 - 25 if y1 > 30 else y1 + 20
                
                # Etiket arka planı
                label_size = cv2.getTextSize(region_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(result, (x1, label_y - 15), (x1 + label_size[0] + 5, label_y + 5), 
                             (0, 0, 0), -1)
                
                # Etiket metni
                cv2.putText(result, region_display, (x1 + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
                
                # Skor göstergesi
                score_text = f"{region_analysis.overall_score:.0f}"
                cv2.putText(result, score_text, (x2 - 25, y2 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Şeffaf birleştirme
        alpha = 0.4
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        return result
    
    def draw_analysis_tables(self, image: np.ndarray, analysis: Dict[str, BodyRegionAnalysis], 
                           recommendations: List[str]) -> np.ndarray:
        """Analiz tablolarını çiz"""
        h, w = image.shape[:2]
        table_width = 400
        
        # Ana tablo arka planı
        cv2.rectangle(image, (w - table_width - 10, 10), (w - 10, h - 10), 
                     self.colors['background'], -1)
        cv2.rectangle(image, (w - table_width - 10, 10), (w - 10, h - 10), 
                     self.colors['border'], 2)
        
        # Başlık
        cv2.putText(image, "PROFESYONEL VUCUT ANALIZI", (w - table_width + 10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.colors['text'], 2)
        
        y = 70
        
        # Genel skorlar tablosu
        muscle_scores = [region.muscle_mass for region in analysis.values()]
        fat_scores = [region.fat_percentage for region in analysis.values()]
        overall_scores = [region.overall_score for region in analysis.values()]
        
        cv2.putText(image, "=== GENEL SKORLAR ===", (w - table_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        
        cv2.putText(image, f"Ortalama Kas Kitlesi:  {np.mean(muscle_scores):.1f}/100", 
                   (w - table_width + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)
        y += 20
        
        cv2.putText(image, f"Ortalama Yag Orani:    {np.mean(fat_scores):.1f}/100", 
                   (w - table_width + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        y += 20
        
        cv2.putText(image, f"Genel Kompozisyon:     {np.mean(overall_scores):.1f}/100", 
                   (w - table_width + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
        y += 35
        
        # Detaylı bölge tablosu
        cv2.putText(image, "=== BOLGE DETAYLARI ===", (w - table_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        
        # Tablo başlıkları
        cv2.putText(image, "Bolge               Kas  Yag  Genel", 
                   (w - table_width + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 20
        
        cv2.line(image, (w - table_width + 15, y), (w - 20, y), (100, 100, 100), 1)
        y += 10
        
        # Bölge detayları
        for region_name, region_analysis in analysis.items():
            if y > h - 80:
                break
            
            region_display = self.anatomical_regions[region_name][:15]
            muscle_score = region_analysis.muscle_mass
            fat_score = region_analysis.fat_percentage
            overall_score = region_analysis.overall_score
            
            # Bölge adı
            cv2.putText(image, f"{region_display[:15]:<15}", (w - table_width + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.colors['text'], 1)
            
            # Kas skoru
            muscle_color = (0, 255, 0) if muscle_score > 70 else (0, 255, 255) if muscle_score > 50 else (0, 100, 255)
            cv2.putText(image, f"{muscle_score:2.0f}", (w - table_width + 240, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, muscle_color, 1)
            
            # Yağ skoru
            fat_color = (0, 255, 0) if fat_score < 30 else (0, 255, 255) if fat_score < 50 else (0, 100, 255)
            cv2.putText(image, f"{fat_score:2.0f}", (w - table_width + 275, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, fat_color, 1)
            
            # Genel skor
            overall_color = region_analysis.status_color
            cv2.putText(image, f"{overall_score:2.0f}", (w - table_width + 315, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, overall_color, 1)
            
            # Durum göstergesi (renkli nokta)
            cv2.circle(image, (w - 30, y - 3), 4, region_analysis.status_color, -1)
            
            y += 18
        
        # Öneriler bölümü
        y += 20
        cv2.putText(image, "=== PROFESYONEL ONERILER ===", (w - table_width + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        y += 25
        
        for i, rec in enumerate(recommendations[:6]):  # Maksimum 6 öneri
            if y > h - 20:
                break
            cv2.putText(image, f"{i+1}. {rec[:38]}", (w - table_width + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 150), 1)
            y += 16
        
        # Renk açıklaması
        y = h - 60
        cv2.putText(image, "Renk Kodlari:", (w - table_width + 15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 15
        
        legend_items = [
            ("Mukemmel (85+)", self.colors['excellent']),
            ("Iyi (65-84)", self.colors['good']),
            ("Orta (50-64)", self.colors['average']),
            ("Zayif (<50)", self.colors['poor'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_offset = (i % 2) * 180
            y_offset = (i // 2) * 15
            cv2.circle(image, (w - table_width + 20 + x_offset, y + y_offset), 4, color, -1)
            cv2.putText(image, label, (w - table_width + 30 + x_offset, y + y_offset + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, self.colors['text'], 1)
        
        return image
    
    def analyze_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, BodyRegionAnalysis], List[str]]:
        """Ana analiz fonksiyonu"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Fotoğraf yüklenemedi: {image_path}")
        
        print(f"🔬 Profesyonel analiz başlatılıyor: {image_path}")
        print("=" * 70)
        
        # MediaPipe ile pose detection
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4
        ) as pose:
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if results.pose_landmarks:
                print("✅ Pose landmarks başarıyla tespit edildi")
                regions = self.get_precise_body_regions(results.pose_landmarks.landmark, image.shape)
                
                # İskelet çizimi
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                )
            else:
                print("⚠️  Pose tespit edilemedi, varsayılan bölgeler kullanılıyor")
                regions = self._get_default_precise_regions(image.shape)
        
        # Bölge bazlı analiz
        analysis_results = {}
        
        print("\n🧠 Bölge bazlı AI analizi...")
        for region_name, coords in regions.items():
            x1, y1, x2, y2 = coords
            roi = image[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Kas analizi
                muscle_mass, definition, _ = self.advanced_muscle_analysis(roi, region_name)
                
                # Yağ analizi
                fat_percentage = self.advanced_fat_analysis(roi, region_name)
                
                # Genel skor hesaplama
                overall_score = (muscle_mass * 0.6 + definition * 0.2 + (100 - fat_percentage) * 0.2)
                
                # Durum rengi belirleme
                status_color = self.determine_status_color(overall_score)
                
                # Bölge özel öneriler
                region_recommendations = []
                if overall_score < 50:
                    region_recommendations.append(f"{self.anatomical_regions[region_name]} odaklı antrenman")
                elif overall_score > 80:
                    region_recommendations.append(f"{self.anatomical_regions[region_name]} maintenance fazı")
                
                analysis_results[region_name] = BodyRegionAnalysis(
                    muscle_mass=muscle_mass,
                    fat_percentage=fat_percentage,
                    definition=definition,
                    overall_score=overall_score,
                    status_color=status_color,
                    recommendations=region_recommendations
                )
                
                print(f"   {self.anatomical_regions[region_name][:20]:<20}: "
                      f"Kas={muscle_mass:4.1f} Yağ={fat_percentage:4.1f} Genel={overall_score:4.1f}")
        
        # Profesyonel öneriler
        recommendations = self.generate_professional_recommendations(analysis_results)
        
        # Görselleştirme
        print("\n🎨 Profesyonel görselleştirme hazırlanıyor...")
        image = self.draw_professional_visualization(image, regions, analysis_results)
        image = self.draw_analysis_tables(image, analysis_results, recommendations)
        
        return image, analysis_results, recommendations

def main():
    if len(sys.argv) != 2:
        print("🏆 PROFESYONEL VÜCUT KOMPOZİSYON ANALİZİ")
        print("=" * 60)
        print("Kullanım:")
        print(f"  python {sys.argv[0]} <foto_yolu>")
        print("\n🔬 Özellikler:")
        print("  • 10 anatomik bölge analizi")
        print("  • AI tabanlı kas-yağ tespiti")  
        print("  • Profesyonel renkli görselleştirme")
        print("  • Detaylı skor tabloları")
        print("  • Uzman seviyesi öneriler")
        return
    
    analyzer = ProfessionalBodyAnalyzer()
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    try:
        result_image, analysis, recommendations = analyzer.analyze_image(image_path)
        
        # Sonucu kaydet
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"professional_analysis_{name}{ext}"
        cv2.imwrite(output_path, result_image)
        
        print(f"\n✅ ANALIZ TAMAMLANDI!")
        print(f"💾 Sonuç dosyası: {output_path}")
        
        # Özet rapor
        muscle_scores = [region.muscle_mass for region in analysis.values()]
        fat_scores = [region.fat_percentage for region in analysis.values()]
        overall_scores = [region.overall_score for region in analysis.values()]
        
        print(f"\n📊 ÖZET RAPOR:")
        print(f"   Ortalama Kas Kitlesi:    {np.mean(muscle_scores):.1f}/100")
        print(f"   Ortalama Yağ Oranı:      {np.mean(fat_scores):.1f}/100")
        print(f"   Genel Kompozisyon:       {np.mean(overall_scores):.1f}/100")
        
        # En iyi ve en zayıf bölgeler
        best_region = max(analysis.items(), key=lambda x: x[1].overall_score)
        worst_region = min(analysis.items(), key=lambda x: x[1].overall_score)
        
        print(f"\n🏆 En güçlü bölge: {analyzer.anatomical_regions[best_region[0]]} ({best_region[1].overall_score:.1f}/100)")
        print(f"🎯 Gelişim alanı: {analyzer.anatomical_regions[worst_region[0]]} ({worst_region[1].overall_score:.1f}/100)")
        
        print(f"\n💡 PROFESYONEL ÖNERİLER:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
            
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()