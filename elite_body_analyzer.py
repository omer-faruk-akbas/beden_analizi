import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
from scipy import ndimage
from skimage import filters, feature, measure, segmentation
import warnings
warnings.filterwarnings("ignore")

@dataclass
class EliteBodyRegionAnalysis:
    """Elite seviye vücut bölgesi analiz sonucu"""
    muscle_mass: float          # Kas kitlesi (0-100)
    muscle_definition: float    # Kas tanımı (0-100)
    fat_percentage: float       # Yağ oranı (0-100)
    vascularity: float         # Damar belirginliği (0-100)
    symmetry: float            # Simetri (0-100)
    overall_score: float       # Genel skor (0-100)
    status_color: Tuple[int, int, int]  # BGR renk kodu
    grade: str                 # A+, A, B+, B, C+, C, D+, D, F
    recommendations: List[str]

class EliteBodyAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Elite renk paleti
        self.colors = {
            'elite': (0, 255, 0),           # Parlak yeşil - A+ (95-100)
            'excellent': (50, 255, 50),     # Yeşil - A (90-94)
            'very_good': (0, 255, 150),     # Açık yeşil - B+ (85-89)
            'good': (0, 255, 255),          # Sarı - B (80-84)
            'above_avg': (0, 200, 255),     # Açık turuncu - C+ (70-79)
            'average': (0, 165, 255),       # Turuncu - C (60-69)
            'below_avg': (0, 100, 255),     # Açık kırmızı - D+ (50-59)
            'poor': (0, 50, 255),           # Kırmızı - D (40-49)
            'very_poor': (0, 0, 255),       # Koyu kırmızı - F (0-39)
            'background': (30, 30, 30),     # Koyu gri
            'text': (255, 255, 255),        # Beyaz
            'border': (150, 150, 150)       # Gri
        }
        
        # Profesyonel bölge tanımları
        self.elite_regions = {
            'upper_pectoralis_major': 'Üst Pektoralis Major',
            'lower_pectoralis_major': 'Alt Pektoralis Major', 
            'upper_rectus_abdominis': 'Üst Rektus Abdominis',
            'lower_rectus_abdominis': 'Alt Rektus Abdominis',
            'external_obliques': 'Eksternal Oblik',
            'serratus_anterior': 'Serratus Anterior',
            'anterior_deltoid': 'Anterior Deltoid',
            'medial_deltoid': 'Medial Deltoid',
            'biceps_brachii': 'Biceps Brachii',
            'triceps_brachii': 'Triceps Brachii',
            'latissimus_dorsi': 'Latissimus Dorsi (Visible)',
            'intercostals': 'İnterkostal Kaslar'
        }
        
        # Grade sistemi
        self.grade_system = {
            (95, 100): ('A+', 'Elite Bodybuilder'),
            (90, 94): ('A', 'Professional Level'),
            (85, 89): ('B+', 'Advanced Athlete'),
            (80, 84): ('B', 'Intermediate-Advanced'),
            (70, 79): ('C+', 'Intermediate'),
            (60, 69): ('C', 'Beginner-Intermediate'),
            (50, 59): ('D+', 'Beginner'),
            (40, 49): ('D', 'Novice'),
            (0, 39): ('F', 'Untrained')
        }
    
    def elite_muscle_analysis(self, roi: np.ndarray, region_name: str) -> Tuple[float, float, float]:
        """Elite seviye kas analizi"""
        if roi.size == 0:
            return 0.0, 0.0, 0.0
        
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 1. MUSCLE FIBER DETECTION (Kas lifi tespiti)
        fiber_score = self._detect_muscle_fibers(gray, region_name)
        
        # 2. MUSCLE SEPARATION (Kas ayrımı)
        separation_score = self._analyze_muscle_separation(gray, region_name)
        
        # 3. MUSCLE DENSITY (Kas yoğunluğu)
        density_score = self._analyze_muscle_density(gray)
        
        # 4. MUSCLE VOLUME (3D kas hacmi)
        volume_score = self._analyze_muscle_volume_advanced(gray)
        
        # 5. STRIATIONS (Kas çizgileri)
        striation_score = self._detect_striations(gray, region_name)
        
        # Bölgeye özel ağırlıklandırma
        weights = self._get_elite_weights(region_name)
        
        # MUSCLE MASS hesaplama
        muscle_mass = (
            fiber_score * weights['fibers'] +
            separation_score * weights['separation'] +
            density_score * weights['density'] +
            volume_score * weights['volume']
        )
        
        # MUSCLE DEFINITION hesaplama
        definition = (
            striation_score * 0.4 +
            separation_score * 0.35 +
            fiber_score * 0.25
        )
        
        # VASCULARITY tespiti
        vascularity = self._detect_vascularity(gray)
        
        return (
            min(100, max(0, muscle_mass)),
            min(100, max(0, definition)),
            min(100, max(0, vascularity))
        )
    
    def _detect_muscle_fibers(self, gray: np.ndarray, region_name: str) -> float:
        """Kas liflerini tespit et"""
        # Gabor filtreleri ile kas lifi yönelimini tespit et
        fiber_responses = []
        
        # Farklı açılarda Gabor filtreleri
        angles = [0, 30, 60, 90, 120, 150]  # Kas lifi yönelimleri
        
        for angle in angles:
            theta = np.pi * angle / 180.0
            kernel = cv2.getGaborKernel((21, 21), 4, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            fiber_responses.append(np.mean(response))
        
        # En güçlü yönelimi bul
        max_response = max(fiber_responses)
        fiber_coherence = np.std(fiber_responses) / (np.mean(fiber_responses) + 1e-6)
        
        # Bölgeye özel kas lifi beklentisi
        expected_fiber_strength = {
            'upper_rectus_abdominis': 0.8,  # Karın kasları belirgin lif
            'lower_rectus_abdominis': 0.8,
            'serratus_anterior': 0.9,       # En belirgin lif yapısı
            'upper_pectoralis_major': 0.6,
            'lower_pectoralis_major': 0.7,
            'external_obliques': 0.5
        }.get(region_name, 0.6)
        
        # Fiber score hesaplama
        fiber_score = (
            (max_response / 255.0) * 60 +              # Lif gücü
            min(fiber_coherence * 200, 40)             # Lif düzeni
        ) * (1.0 + expected_fiber_strength)
        
        return min(100, fiber_score)
    
    def _analyze_muscle_separation(self, gray: np.ndarray, region_name: str) -> float:
        """Kas ayrımını analiz et"""
        # Watershed segmentasyonu ile kas gruplarını ayır
        
        # Threshold ile ön işleme
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Local maxima bul (kas tepeleri)
        local_maxima = feature.peak_local_maxima(dist_transform.flatten(), 
                                               min_distance=5, 
                                               threshold_abs=0.3 * dist_transform.max())
        
        # Kas ayrımı skoru
        num_peaks = len(local_maxima[0])
        expected_peaks = {
            'upper_rectus_abdominis': 2,    # 2 pack
            'lower_rectus_abdominis': 4,    # 4 pack 
            'upper_pectoralis_major': 1,    # Tek kütle
            'lower_pectoralis_major': 1,
            'serratus_anterior': 6,         # Çoklu çıkıntı
            'external_obliques': 2
        }.get(region_name, 1)
        
        # Gradient magnitude (kas sınırları)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        separation_score = (
            min(num_peaks / expected_peaks * 50, 50) +      # Peak sayısı
            np.mean(gradient_magnitude) / 255.0 * 30 +      # Sınır netliği
            np.std(dist_transform) / np.mean(dist_transform) * 20  # Derinlik varyasyonu
        )
        
        return min(100, separation_score)
    
    def _analyze_muscle_density(self, gray: np.ndarray) -> float:
        """Kas yoğunluğunu analiz et"""
        # Local Binary Pattern ile doku analizi
        
        # Simplified LBP
        lbp_image = np.zeros_like(gray)
        padded = np.pad(gray, 1, mode='edge')
        
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                center = padded[i, j]
                neighbors = [
                    padded[i-1, j-1], padded[i-1, j], padded[i-1, j+1],
                    padded[i, j+1], padded[i+1, j+1], padded[i+1, j],
                    padded[i+1, j-1], padded[i, j-1]
                ]
                
                binary_pattern = 0
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary_pattern |= (1 << k)
                
                lbp_image[i-1, j-1] = binary_pattern
        
        # LBP histogram analizi
        hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        
        # Uniform patterns (kas dokusu göstergesi)
        uniform_patterns = 0
        for i in range(256):
            transitions = 0
            binary = format(i, '08b')
            for j in range(8):
                if binary[j] != binary[(j + 1) % 8]:
                    transitions += 1
            if transitions <= 2:
                uniform_patterns += hist[i]
        
        # Density score
        density_score = (
            uniform_patterns * 60 +                    # Uniform doku
            (1 - np.sum(hist**2)) * 40                 # Histogram çeşitliliği
        )
        
        return min(100, density_score)
    
    def _analyze_muscle_volume_advanced(self, gray: np.ndarray) -> float:
        """Gelişmiş kas hacmi analizi"""
        # Shape from shading yaklaşımı
        
        # Gaussian kernel ile smoothing
        smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Gradient vektörleri
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude ve direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Hessian matrix (2nd derivatives)
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
        
        # Curvature hesaplama
        curvature = np.abs(grad_xx + grad_yy)
        
        # Volume indicators
        highlight_regions = gray > (np.mean(gray) + np.std(gray))
        shadow_regions = gray < (np.mean(gray) - np.std(gray))
        
        volume_score = (
            np.mean(magnitude) / 255.0 * 30 +          # Gradient strength
            np.mean(curvature) / 1000.0 * 25 +         # Surface curvature
            np.sum(highlight_regions) / gray.size * 25 + # Highlight ratio
            np.sum(shadow_regions) / gray.size * 20    # Shadow ratio
        )
        
        return min(100, volume_score)
    
    def _detect_striations(self, gray: np.ndarray, region_name: str) -> float:
        """Kas çizgilerini (striations) tespit et"""
        # Çok ince kas çizgileri için özel filtreler
        
        # Ridge detection
        ridge_filter = filters.meijering(gray, sigmas=[1, 2], black_ridges=False)
        
        # Line detection with different angles
        line_responses = []
        for angle in [0, 45, 90, 135]:  # Farklı çizgi yönelimleri
            theta = np.pi * angle / 180.0
            # Custom line detection kernel
            kernel_size = 7
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    x, y = i - center, j - center
                    rotated_x = x * np.cos(theta) - y * np.sin(theta)
                    if abs(rotated_x) < 1:  # Line width
                        kernel[i, j] = 1
            
            kernel = kernel / np.sum(kernel)  # Normalize
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            line_responses.append(np.mean(response))
        
        # En güçlü çizgi yönelimi
        max_line_response = max(line_responses)
        
        # Bölgeye özel striation beklentisi
        expected_striations = {
            'upper_rectus_abdominis': 0.7,
            'lower_rectus_abdominis': 0.8,
            'serratus_anterior': 0.9,      # En belirgin striations
            'upper_pectoralis_major': 0.4,
            'lower_pectoralis_major': 0.5,
            'anterior_deltoid': 0.3,
            'biceps_brachii': 0.5,
            'triceps_brachii': 0.6
        }.get(region_name, 0.4)
        
        striation_score = (
            np.mean(ridge_filter) / 255.0 * 50 +       # Ridge detection
            max_line_response / 255.0 * 50             # Line detection
        ) * (1.0 + expected_striations)
        
        return min(100, striation_score)
    
    def _detect_vascularity(self, gray: np.ndarray) -> float:
        """Damar belirginliğini tespit et"""
        # Vessel-like structure detection
        
        # Frangi vesselness filter simulation
        vessel_responses = []
        
        for sigma in [1, 2, 3]:  # Farklı damar kalınlıkları
            # Hessian eigenvalues approximation
            gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
            
            # Second derivatives
            Lxx = cv2.Sobel(gaussian, cv2.CV_64F, 2, 0, ksize=3)
            Lyy = cv2.Sobel(gaussian, cv2.CV_64F, 0, 2, ksize=3)
            Lxy = cv2.Sobel(gaussian, cv2.CV_64F, 1, 1, ksize=3)
            
            # Approximate eigenvalues
            lambda1 = 0.5 * (Lxx + Lyy + np.sqrt((Lxx - Lyy)**2 + 4*Lxy**2))
            lambda2 = 0.5 * (Lxx + Lyy - np.sqrt((Lxx - Lyy)**2 + 4*Lxy**2))
            
            # Vesselness measure
            vesselness = np.exp(-lambda1**2 / (2 * 0.5**2)) * (1 - np.exp(-lambda2**2 / (2 * 0.5**2)))
            vesselness[lambda1 < 0] = 0  # Only tube-like structures
            
            vessel_responses.append(np.mean(vesselness))
        
        # Dark line detection (veins)
        dark_lines = gray < (np.mean(gray) - 1.5 * np.std(gray))
        thin_dark_regions = cv2.morphologyEx(dark_lines.astype(np.uint8), 
                                           cv2.MORPH_OPEN, 
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        
        vascularity_score = (
            max(vessel_responses) * 60 +                # Vessel detection
            np.sum(thin_dark_regions) / gray.size * 40  # Dark line ratio
        )
        
        return min(100, vascularity_score)
    
    def _get_elite_weights(self, region_name: str) -> Dict[str, float]:
        """Elite seviye bölge ağırlıkları"""
        weights = {
            'upper_rectus_abdominis': {
                'fibers': 0.3, 'separation': 0.35, 'density': 0.2, 'volume': 0.15
            },
            'lower_rectus_abdominis': {
                'fibers': 0.35, 'separation': 0.4, 'density': 0.15, 'volume': 0.1
            },
            'serratus_anterior': {
                'fibers': 0.4, 'separation': 0.35, 'density': 0.15, 'volume': 0.1
            },
            'upper_pectoralis_major': {
                'fibers': 0.25, 'separation': 0.2, 'density': 0.25, 'volume': 0.3
            },
            'lower_pectoralis_major': {
                'fibers': 0.3, 'separation': 0.25, 'density': 0.25, 'volume': 0.2
            },
            'external_obliques': {
                'fibers': 0.35, 'separation': 0.3, 'density': 0.2, 'volume': 0.15
            },
            'anterior_deltoid': {
                'fibers': 0.2, 'separation': 0.15, 'density': 0.3, 'volume': 0.35
            },
            'biceps_brachii': {
                'fibers': 0.25, 'separation': 0.2, 'density': 0.25, 'volume': 0.3
            },
            'triceps_brachii': {
                'fibers': 0.3, 'separation': 0.25, 'density': 0.25, 'volume': 0.2
            }
        }
        return weights.get(region_name, {
            'fibers': 0.25, 'separation': 0.25, 'density': 0.25, 'volume': 0.25
        })
    
    def advanced_fat_analysis(self, roi: np.ndarray, region_name: str) -> float:
        """Gelişmiş yağ analizi"""
        if roi.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 1. SUBCUTANEOUS FAT DETECTION
        subcutaneous_fat = self._detect_subcutaneous_fat(gray)
        
        # 2. SKIN FOLD ANALYSIS
        skin_fold = self._analyze_skin_folds(gray)
        
        # 3. SURFACE SMOOTHNESS
        smoothness = self._analyze_surface_smoothness(gray)
        
        # 4. FAT LAYER THICKNESS
        thickness = self._estimate_fat_thickness(gray)
        
        # Bölgeye özel yağ ağırlıkları
        fat_weights = {
            'upper_pectoralis_major': [0.3, 0.2, 0.3, 0.2],
            'lower_pectoralis_major': [0.35, 0.25, 0.25, 0.15],
            'upper_rectus_abdominis': [0.25, 0.3, 0.25, 0.2],
            'lower_rectus_abdominis': [0.3, 0.35, 0.2, 0.15],
            'external_obliques': [0.4, 0.3, 0.2, 0.1]  # Love handles area
        }.get(region_name, [0.3, 0.25, 0.25, 0.2])
        
        fat_score = (
            subcutaneous_fat * fat_weights[0] +
            skin_fold * fat_weights[1] +
            smoothness * fat_weights[2] +
            thickness * fat_weights[3]
        )
        
        return min(100, max(0, fat_score))
    
    def _detect_subcutaneous_fat(self, gray: np.ndarray) -> float:
        """Deri altı yağını tespit et"""
        # Brightness analysis for fat tissue
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Fat tissue typically has:
        # 1. Higher brightness values
        # 2. Lower contrast (more homogeneous)
        # 3. Smoother texture
        
        # Brightness component
        brightness_score = 0
        if mean_brightness > 140:
            brightness_score = min((mean_brightness - 140) / 115 * 100, 100)
        
        # Homogeneity (low contrast = fat)
        homogeneity_score = max(0, 100 - brightness_std / 50 * 100)
        
        # Texture smoothness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        smoothness_score = max(0, 100 - min(laplacian_var / 500 * 100, 100))
        
        subcutaneous_score = (
            brightness_score * 0.4 +
            homogeneity_score * 0.35 +
            smoothness_score * 0.25
        )
        
        return subcutaneous_score
    
    def _analyze_skin_folds(self, gray: np.ndarray) -> float:
        """Deri kıvrımlarını analiz et"""
        # Skin folds indicate fat accumulation
        
        # Edge detection for folds
        edges = cv2.Canny(gray, 30, 80)
        
        # Morphological operations to detect fold-like structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
        fold_enhanced = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Count fold pixels
        fold_ratio = np.sum(fold_enhanced > 0) / fold_enhanced.size
        
        # Curved line detection (folds are usually curved)
        contours, _ = cv2.findContours(fold_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        curved_folds = 0
        for contour in contours:
            if len(contour) > 5:
                # Fit ellipse to detect curved structures
                try:
                    ellipse = cv2.fitEllipse(contour)
                    # If ellipse aspect ratio indicates curve
                    aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
                    if 2 < aspect_ratio < 10:  # Curved fold characteristics
                        curved_folds += 1
                except:
                    pass
        
        fold_score = (
            fold_ratio * 100 * 0.6 +           # Fold pixel ratio
            min(curved_folds * 20, 40)         # Curved fold count
        )
        
        return min(100, fold_score)
    
    def _analyze_surface_smoothness(self, gray: np.ndarray) -> float:
        """Yüzey düzgünlüğünü analiz et"""
        # High smoothness = more fat coverage
        
        # Gaussian blur difference
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        difference = np.abs(gray.astype(float) - blurred.astype(float))
        smoothness_metric = 100 - np.mean(difference) / 255 * 100
        
        # Standard deviation (low = smooth = fat)
        local_std = []
        kernel_size = 5
        for i in range(0, gray.shape[0] - kernel_size, kernel_size):
            for j in range(0, gray.shape[1] - kernel_size, kernel_size):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                local_std.append(np.std(patch))
        
        avg_local_std = np.mean(local_std)
        std_smoothness = max(0, 100 - avg_local_std / 30 * 100)
        
        surface_smoothness = (smoothness_metric * 0.6 + std_smoothness * 0.4)
        
        return surface_smoothness
    
    def _estimate_fat_thickness(self, gray: np.ndarray) -> float:
        """Yağ tabakası kalınlığını tahmin et"""
        # Analyze intensity profiles to estimate fat layer thickness
        
        # Horizontal and vertical intensity profiles
        h_profile = np.mean(gray, axis=0)
        v_profile = np.mean(gray, axis=1)
        
        # Look for plateau regions (fat layers)
        def find_plateaus(profile):
            gradient = np.gradient(profile)
            low_gradient = np.abs(gradient) < np.std(gradient) * 0.5
            
            plateau_lengths = []
            current_length = 0
            
            for is_plateau in low_gradient:
                if is_plateau:
                    current_length += 1
                else:
                    if current_length > 0:
                        plateau_lengths.append(current_length)
                    current_length = 0
            
            return plateau_lengths
        
        h_plateaus = find_plateaus(h_profile)
        v_plateaus = find_plateaus(v_profile)
        
        # Fat thickness indicators
        avg_plateau_h = np.mean(h_plateaus) if h_plateaus else 0
        avg_plateau_v = np.mean(v_plateaus) if v_plateaus else 0
        
        # Brightness uniformity (thick fat = uniform brightness)
        brightness_uniformity = 100 - cv2.Laplacian(gray, cv2.CV_64F).var() / 1000 * 100
        brightness_uniformity = max(0, min(brightness_uniformity, 100))
        
        thickness_score = (
            min(avg_plateau_h / 20 * 40, 40) +     # Horizontal plateau
            min(avg_plateau_v / 20 * 30, 30) +     # Vertical plateau  
            brightness_uniformity * 0.3             # Uniformity
        )
        
        return min(100, thickness_score)
    
    def get_elite_body_regions(self, landmarks, img_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
        """Elite seviye hassas bölge koordinatları"""
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
            nose = landmarks[0]
            
            # Koordinat dönüşümü
            ls = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            rs = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            lh = (int(left_hip.x * w), int(left_hip.y * h))
            rh = (int(right_hip.x * w), int(right_hip.y * h))
            le = (int(left_elbow.x * w), int(left_elbow.y * h))
            re = (int(right_elbow.x * w), int(right_elbow.y * h))
            nose_pos = (int(nose.x * w), int(nose.y * h))
            
            # Torso geometrisi
            torso_center_x = (ls[0] + rs[0]) // 2
            torso_width = abs(rs[0] - ls[0])
            torso_height = abs((lh[1] + rh[1]) // 2 - (ls[1] + rs[1]) // 2)
            
            # ÜST PEKTORALIS MAJOR (Clavicular portion)
            chest_top = max(10, (ls[1] + rs[1]) // 2 - 30)
            chest_upper_bottom = (ls[1] + rs[1]) // 2 + 25
            regions['upper_pectoralis_major'] = (
                torso_center_x - int(torso_width * 0.35),
                chest_top,
                torso_center_x + int(torso_width * 0.35),
                chest_upper_bottom
            )
            
            # ALT PEKTORALIS MAJOR (Sternal portion)
            chest_lower_top = chest_upper_bottom - 15
            chest_lower_bottom = chest_lower_top + int(torso_width * 0.4)
            regions['lower_pectoralis_major'] = (
                torso_center_x - int(torso_width * 0.45),
                chest_lower_top,
                torso_center_x + int(torso_width * 0.45),
                chest_lower_bottom
            )
            
            # ÜST REKTUS ABDOMİNİS (Upper abs - 2&4 pack)
            abs_top = chest_lower_bottom - 10
            abs_upper_bottom = abs_top + int(torso_height * 0.25)
            regions['upper_rectus_abdominis'] = (
                torso_center_x - int(torso_width * 0.22),
                abs_top,
                torso_center_x + int(torso_width * 0.22),
                abs_upper_bottom
            )
            
            # ALT REKTUS ABDOMİNİS (Lower abs - 6&8 pack)
            abs_lower_top = abs_upper_bottom - 5
            abs_lower_bottom = min(h - 50, (lh[1] + rh[1]) // 2 - 10)
            regions['lower_rectus_abdominis'] = (
                torso_center_x - int(torso_width * 0.2),
                abs_lower_top,
                torso_center_x + int(torso_width * 0.2),
                abs_lower_bottom
            )
            
            # EKSTERNAl OBLİK (Love handles bölgesi)
            oblique_top = abs_top
            oblique_bottom = abs_lower_bottom
            regions['external_obliques'] = (
                ls[0] - int(torso_width * 0.3),
                oblique_top,
                ls[0] - 10,
                oblique_bottom
            )
            
            # SERRATUS ANTERİOR (Pilot wings)
            serratus_top = (ls[1] + rs[1]) // 2 + 15
            serratus_bottom = serratus_top + int(torso_width * 0.35)
            regions['serratus_anterior'] = (
                ls[0] - 25,
                serratus_top,
                ls[0] + 15,
                serratus_bottom
            )
            
            # ANTERİOR DELTOİD (Ön omuz)
            deltoid_size = int(torso_width * 0.18)
            regions['anterior_deltoid'] = (
                ls[0] - deltoid_size // 2,
                ls[1] - 25,
                ls[0] + deltoid_size // 2,
                ls[1] + 45
            )
            
            # MEDİAL DELTOİD (Yan omuz)
            regions['medial_deltoid'] = (
                ls[0] - deltoid_size - 15,
                ls[1] - 15,
                ls[0] - 15,
                ls[1] + 55
            )
            
            # BİCEPS BRACHİİ
            arm_width = abs(le[0] - ls[0]) // 2
            regions['biceps_brachii'] = (
                le[0] - arm_width // 2,
                (ls[1] + le[1]) // 2 - 30,
                le[0] + arm_width // 2,
                (ls[1] + le[1]) // 2 + 30
            )
            
            # TRİCEPS BRACHİİ
            regions['triceps_brachii'] = (
                le[0] - arm_width // 2 + 15,
                (ls[1] + le[1]) // 2 - 25,
                le[0] + arm_width // 2 + 15,
                (ls[1] + le[1]) // 2 + 35
            )
            
            # İNTERKOSTAL KASLAR (Kaburga arası)
            regions['intercostals'] = (
                torso_center_x - int(torso_width * 0.15),
                chest_lower_bottom - 20,
                torso_center_x + int(torso_width * 0.15),
                abs_top + 20
            )
            
        except Exception as e:
            print(f"⚠️  Elite landmark hatası: {e}")
            regions = self._get_default_elite_regions(img_shape)
        
        # Sınır kontrolü ve optimizasyon
        for region_name, (x1, y1, x2, y2) in regions.items():
            regions[region_name] = (
                max(0, min(x1, w-1)),
                max(0, min(y1, h-1)),
                max(0, min(x2, w-1)),
                max(0, min(y2, h-1))
            )
        
        return regions
    
    def _get_default_elite_regions(self, img_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
        """Varsayılan elite bölgeler"""
        h, w = img_shape[:2]
        return {
            'upper_pectoralis_major': (int(w*0.32), int(h*0.12), int(w*0.68), int(h*0.25)),
            'lower_pectoralis_major': (int(w*0.28), int(h*0.22), int(w*0.72), int(h*0.4)),
            'upper_rectus_abdominis': (int(w*0.42), int(h*0.35), int(w*0.58), int(h*0.5)),
            'lower_rectus_abdominis': (int(w*0.43), int(h*0.47), int(w*0.57), int(h*0.65)),
            'external_obliques': (int(w*0.15), int(h*0.35), int(w*0.35), int(h*0.6)),
            'serratus_anterior': (int(w*0.18), int(h*0.25), int(w*0.32), int(h*0.45)),
            'anterior_deltoid': (int(w*0.15), int(h*0.08), int(w*0.3), int(h*0.22)),
            'medial_deltoid': (int(w*0.08), int(h*0.1), int(w*0.22), int(h*0.28)),
            'biceps_brachii': (int(w*0.1), int(h*0.22), int(w*0.2), int(h*0.4)),
            'triceps_brachii': (int(w*0.12), int(h*0.24), int(w*0.22), int(h*0.42)),
            'intercostals': (int(w*0.4), int(h*0.28), int(w*0.6), int(h*0.42))
        }
    
    def determine_grade_and_color(self, overall_score: float) -> Tuple[str, str, Tuple[int, int, int]]:
        """Skor bazlı grade ve renk belirleme"""
        for (min_score, max_score), (grade, description) in self.grade_system.items():
            if min_score <= overall_score <= max_score:
                if grade == 'A+':
                    color = self.colors['elite']
                elif grade == 'A':
                    color = self.colors['excellent']
                elif grade == 'B+':
                    color = self.colors['very_good']
                elif grade == 'B':
                    color = self.colors['good']
                elif grade == 'C+':
                    color = self.colors['above_avg']
                elif grade == 'C':
                    color = self.colors['average']
                elif grade == 'D+':
                    color = self.colors['below_avg']
                elif grade == 'D':
                    color = self.colors['poor']
                else:  # F
                    color = self.colors['very_poor']
                
                return grade, description, color
        
        return 'F', 'Untrained', self.colors['very_poor']
    
    def generate_elite_recommendations(self, analysis: Dict[str, EliteBodyRegionAnalysis]) -> List[str]:
        """Elite seviye öneriler"""
        recommendations = []
        
        # Skorları topla
        muscle_scores = [region.muscle_mass for region in analysis.values()]
        definition_scores = [region.muscle_definition for region in analysis.values()]
        fat_scores = [region.fat_percentage for region in analysis.values()]
        vascularity_scores = [region.vascularity for region in analysis.values()]
        overall_scores = [region.overall_score for region in analysis.values()]
        
        # Ortalamalar
        avg_muscle = np.mean(muscle_scores)
        avg_definition = np.mean(definition_scores)
        avg_fat = np.mean(fat_scores)
        avg_vascularity = np.mean(vascularity_scores)
        avg_overall = np.mean(overall_scores)
        
        # Grade bazlı öneriler
        grade, description, _ = self.determine_grade_and_color(avg_overall)
        
        if grade in ['A+', 'A']:
            recommendations.append(f"🏆 {description} seviyesi - Mükemmel!")
            recommendations.append("🎯 Contest prep için hazırsın")
            recommendations.append("💧 Hydration ve peak week protokolü")
        elif grade in ['B+', 'B']:
            recommendations.append(f"💪 {description} - İleri seviye gelişim")
            recommendations.append("🔥 Contest prep düşünülebilir")
            recommendations.append("⚡ Weak points üzerinde odaklan")
        elif grade in ['C+', 'C']:
            recommendations.append(f"📈 {description} - İyi ilerleme")
            recommendations.append("🏋️ Progressive overload artır")
            recommendations.append("🍖 Protein timing optimize et")
        else:
            recommendations.append(f"🎯 {description} - Temel güçlendirme")
            recommendations.append("💥 Heavy compound movements")
            recommendations.append("📊 Tracking ve consistency")
        
        # Spesifik analiz önerileri
        if avg_definition < 60:
            recommendations.append("🔪 Cutting phase gerekli")
            recommendations.append("🏃 Cardio protokolü ekle")
        elif avg_definition > 80:
            recommendations.append("💎 Excellent definition!")
            
        if avg_vascularity > 70:
            recommendations.append("🩸 Vascularity excellent - contest ready!")
        elif avg_vascularity < 30:
            recommendations.append("💪 Vascularity için body fat düşür")
        
        # Zayıf bölgeler
        weak_regions = [(name, region) for name, region in analysis.items() 
                       if region.overall_score < 60]
        
        if weak_regions:
            weak_names = [self.elite_regions[name] for name, _ in weak_regions[:2]]
            recommendations.append(f"🎯 Priority: {', '.join(weak_names)}")
        
        # Elite bölgeler
        elite_regions = [(name, region) for name, region in analysis.items() 
                        if region.overall_score > 85]
        
        if elite_regions:
            elite_names = [self.elite_regions[name] for name, _ in elite_regions[:2]]
            recommendations.append(f"✨ Elite regions: {', '.join(elite_names)}")
        
        return recommendations[:8]
    
    def draw_elite_visualization(self, image: np.ndarray, regions: Dict[str, Tuple[int, int, int, int]], 
                                analysis: Dict[str, EliteBodyRegionAnalysis]) -> np.ndarray:
        """Elite seviye görselleştirme"""
        overlay = image.copy()
        result = image.copy()
        
        # Bölgeleri renklendir
        for region_name, (x1, y1, x2, y2) in regions.items():
            if region_name in analysis:
                region_analysis = analysis[region_name]
                
                # Bölgeyi renklendir
                cv2.rectangle(overlay, (x1, y1), (x2, y2), region_analysis.status_color, -1)
                
                # Grade ile kenarlık
                border_thickness = 3 if region_analysis.grade in ['A+', 'A'] else 2
                cv2.rectangle(result, (x1, y1), (x2, y2), region_analysis.status_color, border_thickness)
                
                # Elite grade göstergesi
                if region_analysis.grade in ['A+', 'A']:
                    # Elite crown simgesi
                    cv2.circle(result, (x2 - 15, y1 + 15), 8, (0, 255, 255), -1)
                    cv2.putText(result, region_analysis.grade, (x2 - 22, y1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Bölge etiketi
                region_display = self.elite_regions[region_name][:10]
                label_y = y1 - 30 if y1 > 35 else y1 + 25
                
                # Etiket arka planı
                label_size = cv2.getTextSize(region_display, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(result, (x1, label_y - 18), (x1 + label_size[0] + 8, label_y + 8), 
                             (0, 0, 0), -1)
                
                # Etiket metni
                cv2.putText(result, region_display, (x1 + 3, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
                
                # Skor göstergesi
                score_text = f"{region_analysis.overall_score:.0f}"
                cv2.putText(result, score_text, (x2 - 30, y2 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['text'], 1)
                
                # Vascularity göstergesi
                if region_analysis.vascularity > 60:
                    cv2.circle(result, (x1 + 10, y2 - 10), 3, (0, 0, 255), -1)  # Kırmızı damar göstergesi
        
        # Şeffaf birleştirme
        alpha = 0.35
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        return result
    
    def draw_elite_analysis_panel(self, image: np.ndarray, analysis: Dict[str, EliteBodyRegionAnalysis], 
                                 recommendations: List[str]) -> np.ndarray:
        """Elite analiz paneli"""
        h, w = image.shape[:2]
        panel_width = 450
        
        # Ana panel
        cv2.rectangle(image, (w - panel_width - 10, 10), (w - 10, h - 10), 
                     self.colors['background'], -1)
        cv2.rectangle(image, (w - panel_width - 10, 10), (w - 10, h - 10), 
                     self.colors['border'], 3)
        
        # Elite başlık
        cv2.putText(image, "ELITE BODY COMPOSITION ANALYSIS", (w - panel_width + 15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        y = 70
        
        # Genel grade ve skor
        muscle_scores = [region.muscle_mass for region in analysis.values()]
        definition_scores = [region.muscle_definition for region in analysis.values()]
        fat_scores = [region.fat_percentage for region in analysis.values()]
        vascularity_scores = [region.vascularity for region in analysis.values()]
        overall_scores = [region.overall_score for region in analysis.values()]
        
        avg_overall = np.mean(overall_scores)
        grade, description, grade_color = self.determine_grade_and_color(avg_overall)
        
        # Grade kutusu
        cv2.rectangle(image, (w - panel_width + 15, y), (w - panel_width + 120, y + 40), grade_color, -1)
        cv2.rectangle(image, (w - panel_width + 15, y), (w - panel_width + 120, y + 40), (255, 255, 255), 2)
        cv2.putText(image, f"GRADE: {grade}", (w - panel_width + 20, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(image, f"{avg_overall:.1f}/100", (w - panel_width + 20, y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        # Description
        cv2.putText(image, description, (w - panel_width + 130, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        y += 60
        
        # Elite metrikleri
        cv2.putText(image, "=== ELITE METRICS ===", (w - panel_width + 15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        
        metrics = [
            (f"Muscle Mass:      {np.mean(muscle_scores):.1f}/100", (0, 255, 100)),
            (f"Definition:       {np.mean(definition_scores):.1f}/100", (100, 255, 200)),
            (f"Fat Percentage:   {np.mean(fat_scores):.1f}/100", (100, 100, 255)),
            (f"Vascularity:      {np.mean(vascularity_scores):.1f}/100", (0, 100, 255))
        ]
        
        for metric_text, color in metrics:
            cv2.putText(image, metric_text, (w - panel_width + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 20
        
        y += 15
        
        # Detaylı bölge tablosu
        cv2.putText(image, "=== REGIONAL ANALYSIS ===", (w - panel_width + 15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y += 25
        
        # Tablo başlıkları
        cv2.putText(image, "Region                 Muscle Def  Fat  Grade", 
                   (w - panel_width + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 20
        
        cv2.line(image, (w - panel_width + 20, y), (w - 20, y), (100, 100, 100), 1)
        y += 10
        
        # Bölge detayları
        for region_name, region_analysis in analysis.items():
            if y > h - 120:
                break
            
            region_display = self.elite_regions[region_name][:18]
            muscle_score = region_analysis.muscle_mass
            definition_score = region_analysis.muscle_definition
            fat_score = region_analysis.fat_percentage
            grade = region_analysis.grade
            
            # Bölge adı
            cv2.putText(image, f"{region_display[:18]:<18}", (w - panel_width + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
            
            # Muscle score
            muscle_color = (0, 255, 0) if muscle_score > 80 else (0, 255, 255) if muscle_score > 60 else (0, 100, 255)
            cv2.putText(image, f"{muscle_score:2.0f}", (w - panel_width + 240, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, muscle_color, 1)
            
            # Definition score
            def_color = (0, 255, 0) if definition_score > 80 else (0, 255, 255) if definition_score > 60 else (0, 100, 255)
            cv2.putText(image, f"{definition_score:2.0f}", (w - panel_width + 270, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, def_color, 1)
            
            # Fat score
            fat_color = (0, 255, 0) if fat_score < 30 else (0, 255, 255) if fat_score < 50 else (0, 100, 255)
            cv2.putText(image, f"{fat_score:2.0f}", (w - panel_width + 300, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, fat_color, 1)
            
            # Grade
            cv2.putText(image, grade, (w - panel_width + 340, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, region_analysis.status_color, 1)
            
            # Vascularity göstergesi
            if region_analysis.vascularity > 60:
                cv2.circle(image, (w - panel_width + 380, y - 3), 2, (0, 0, 255), -1)
            
            y += 16
        
        # Elite öneriler
        y += 20
        cv2.putText(image, "=== ELITE RECOMMENDATIONS ===", (w - panel_width + 15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        y += 25
        
        for i, rec in enumerate(recommendations[:6]):
            if y > h - 20:
                break
            cv2.putText(image, f"{i+1}. {rec[:42]}", (w - panel_width + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 150), 1)
            y += 16
        
        # Elite legend
        y = h - 80
        cv2.putText(image, "Elite Color Coding:", (w - panel_width + 20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 15
        
        legend_items = [
            ("A+ Elite", self.colors['elite']),
            ("A Excellent", self.colors['excellent']),
            ("B+ Very Good", self.colors['very_good']),
            ("F Untrained", self.colors['very_poor'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_offset = (i % 2) * 200
            y_offset = (i // 2) * 15
            cv2.circle(image, (w - panel_width + 25 + x_offset, y + y_offset), 4, color, -1)
            cv2.putText(image, label, (w - panel_width + 35 + x_offset, y + y_offset + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, self.colors['text'], 1)
        
        return image
    
    def analyze_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, EliteBodyRegionAnalysis], List[str]]:
        """Elite seviye ana analiz"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Fotoğraf yüklenemedi: {image_path}")
        
        print(f"🏆 ELITE BODY ANALYSIS başlatılıyor: {image_path}")
        print("=" * 80)
        
        # MediaPipe ile pose detection
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # En yüksek complexity
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if results.pose_landmarks:
                print("✅ Elite pose landmarks tespit edildi")
                regions = self.get_elite_body_regions(results.pose_landmarks.landmark, image.shape)
                
                # Elite iskelet çizimi
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
            else:
                print("⚠️  Pose tespit edilemedi, varsayılan elite bölgeler kullanılıyor")
                regions = self._get_default_elite_regions(image.shape)
        
        # Elite bölge analizi
        analysis_results = {}
        
        print("\n🧠 Elite AI analizi başlatılıyor...")
        for region_name, coords in regions.items():
            x1, y1, x2, y2 = coords
            roi = image[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Elite kas analizi
                muscle_mass, definition, vascularity = self.elite_muscle_analysis(roi, region_name)
                
                # Elite yağ analizi  
                fat_percentage = self.advanced_fat_analysis(roi, region_name)
                
                # Simetri analizi (varsayılan)
                symmetry = 85.0  # Basit simetri skoru
                
                # Elite genel skor
                overall_score = (
                    muscle_mass * 0.35 +
                    definition * 0.25 +
                    (100 - fat_percentage) * 0.2 +
                    vascularity * 0.1 +
                    symmetry * 0.1
                )
                
                # Grade ve renk
                grade, description, status_color = self.determine_grade_and_color(overall_score)
                
                # Elite öneriler
                region_recommendations = []
                if grade in ['F', 'D', 'D+']:
                    region_recommendations.append(f"Heavy compound movements for {self.elite_regions[region_name]}")
                elif grade in ['A+', 'A']:
                    region_recommendations.append(f"Contest ready! Maintain {self.elite_regions[region_name]}")
                
                analysis_results[region_name] = EliteBodyRegionAnalysis(
                    muscle_mass=muscle_mass,
                    muscle_definition=definition,
                    fat_percentage=fat_percentage,
                    vascularity=vascularity,
                    symmetry=symmetry,
                    overall_score=overall_score,
                    status_color=status_color,
                    grade=grade,
                    recommendations=region_recommendations
                )
                
                print(f"   {self.elite_regions[region_name][:25]:<25}: "
                      f"M={muscle_mass:4.1f} D={definition:4.1f} F={fat_percentage:4.1f} "
                      f"V={vascularity:4.1f} [{grade}] {overall_score:4.1f}")
        
        # Elite öneriler
        recommendations = self.generate_elite_recommendations(analysis_results)
        
        # Elite görselleştirme
        print("\n🎨 Elite görselleştirme hazırlanıyor...")
        image = self.draw_elite_visualization(image, regions, analysis_results)
        image = self.draw_elite_analysis_panel(image, analysis_results, recommendations)
        
        return image, analysis_results, recommendations

def main():
    if len(sys.argv) != 2:
        print("🏆 ELITE BODY COMPOSITION ANALYZER")
        print("=" * 70)
        print("Kullanım:")
        print(f"  python {sys.argv[0]} <foto_yolu>")
        print("\n🥇 Elite Özellikler:")
        print("  • 12+ anatomik bölge analizi")
        print("  • AI tabanlı kas lifi tespiti")
        print("  • Striations ve vascularity analizi")
        print("  • Professional grade sistemi (A+ to F)")
        print("  • Contest prep değerlendirmesi")
        print("  • Elite seviye görselleştirme")
        return
    
    analyzer = EliteBodyAnalyzer()
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    try:
        result_image, analysis, recommendations = analyzer.analyze_image(image_path)
        
        # Elite sonuç kaydet
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"elite_analysis_{name}{ext}"
        cv2.imwrite(output_path, result_image)
        
        print(f"\n🏆 ELITE ANALYSIS COMPLETED!")
        print(f"💾 Elite sonuç dosyası: {output_path}")
        
        # Elite özet rapor
        muscle_scores = [region.muscle_mass for region in analysis.values()]
        definition_scores = [region.muscle_definition for region in analysis.values()]
        fat_scores = [region.fat_percentage for region in analysis.values()]
        vascularity_scores = [region.vascularity for region in analysis.values()]
        overall_scores = [region.overall_score for region in analysis.values()]
        
        avg_overall = np.mean(overall_scores)
        grade, description, _ = analyzer.determine_grade_and_color(avg_overall)
        
        print(f"\n📊 ELITE SUMMARY:")
        print(f"   Overall Grade:        {grade} ({description})")
        print(f"   Muscle Mass:          {np.mean(muscle_scores):.1f}/100")
        print(f"   Definition:           {np.mean(definition_scores):.1f}/100")
        print(f"   Fat Percentage:       {np.mean(fat_scores):.1f}/100")
        print(f"   Vascularity:          {np.mean(vascularity_scores):.1f}/100")
        print(f"   Composite Score:      {avg_overall:.1f}/100")
        
        # Elite bölgeler
        elite_regions = [(name, region) for name, region in analysis.items() 
                        if region.grade in ['A+', 'A']]
        weak_regions = [(name, region) for name, region in analysis.items() 
                       if region.overall_score < 50]
        
        if elite_regions:
            elite_names = [analyzer.elite_regions[name] for name, _ in elite_regions]
            print(f"\n🏆 Elite Regions: {', '.join(elite_names)}")
        
        if weak_regions:
            weak_names = [analyzer.elite_regions[name] for name, _ in weak_regions]
            print(f"🎯 Development Areas: {', '.join(weak_names)}")
        
        print(f"\n💡 ELITE RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
            
    except Exception as e:
        print(f"❌ Elite analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()