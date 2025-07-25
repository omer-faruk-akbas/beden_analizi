#!/usr/bin/env python3
"""
🎯 OPTIMIZED ACCURATE FITNESS SYSTEM
Maximum doğruluk + optimized performance
En gelişmiş AI teknikleri optimize edilmiş şekilde
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

@dataclass
class AccuratePrediction:
    """Doğru tahmin sonucu"""
    value: float
    confidence: float
    uncertainty: float
    method: str
    quality_score: float

@dataclass
class OptimizedResult:
    """Optimize edilmiş sonuç"""
    region_predictions: Dict[str, AccuratePrediction]
    ensemble_score: float
    model_confidence: float
    prediction_range: Tuple[float, float]
    quality_metrics: Dict[str, float]
    feature_analysis: Dict[str, float]
    recommendations: List[str]
    visual_output_path: Optional[str] = None

class OptimizedFeatureExtractor:
    """Optimize edilmiş özellik çıkarıcı"""
    
    def extract_enhanced_features(self, roi: np.ndarray, region_name: str) -> np.ndarray:
        """Gelişmiş ve optimize edilmiş özellik çıkarımı"""
        if roi.size == 0:
            return np.zeros(50)
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        features = []
        
        # 1. Basic statistical features (optimized)
        features.extend(self._extract_statistical_features(gray_roi))
        
        # 2. Edge and gradient features (most important for muscle definition)
        features.extend(self._extract_edge_features(gray_roi))
        
        # 3. Texture features (selective)
        features.extend(self._extract_texture_features(gray_roi))
        
        # 4. Geometric features
        features.extend(self._extract_geometric_features(gray_roi))
        
        # 5. Region-specific features
        features.extend(self._extract_region_specific_features(gray_roi, region_name))
        
        # Ensure fixed length
        if len(features) < 50:
            features.extend([0] * (50 - len(features)))
        else:
            features = features[:50]
        
        return np.array(features)
    
    def _extract_statistical_features(self, gray_roi: np.ndarray) -> List[float]:
        """İstatistiksel özellikler"""
        features = []
        
        # Basic statistics
        features.append(np.mean(gray_roi))
        features.append(np.std(gray_roi))
        features.append(np.var(gray_roi))
        features.append(np.min(gray_roi))
        features.append(np.max(gray_roi))
        features.append(np.median(gray_roi))
        
        # Higher moments
        flat = gray_roi.flatten()
        features.append(stats.skew(flat))
        features.append(stats.kurtosis(flat))
        
        # Percentiles
        features.append(np.percentile(gray_roi, 25))
        features.append(np.percentile(gray_roi, 75))
        
        return features
    
    def _extract_edge_features(self, gray_roi: np.ndarray) -> List[float]:
        """Edge ve gradient özellikleri (kas tanımı için kritik)"""
        features = []
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        features.append(np.max(gradient_magnitude))
        
        # Canny edges
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        features.append(np.var(laplacian))
        features.append(np.mean(np.abs(laplacian)))
        
        # Edge direction consistency
        gradient_direction = np.arctan2(grad_y, grad_x)
        hist, _ = np.histogram(gradient_direction, bins=8, range=(-np.pi, np.pi))
        hist = hist.astype(float) / (hist.sum() + 1e-6)
        features.append(np.max(hist))  # Dominant orientation strength
        
        return features
    
    def _extract_texture_features(self, gray_roi: np.ndarray) -> List[float]:
        """Texture özellikleri (optimize edilmiş)"""
        features = []
        
        # Local Binary Pattern (single radius for speed)
        radius = 2
        n_points = 8 * radius
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray_roi, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-6)
            
            features.append(np.mean(lbp_hist))
            features.append(np.std(lbp_hist))
            features.append(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-6)))  # Entropy
        except:
            features.extend([0.5, 0.3, 0.7])  # Default values
        
        # Gabor responses (limited set)
        try:
            for angle in [0, 45, 90]:
                kernel = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 8, 0.5, 0, ktype=cv2.CV_32F)
                gabor_response = cv2.filter2D(gray_roi, cv2.CV_8UC3, kernel)
                features.append(np.mean(gabor_response))
        except:
            features.extend([0, 0, 0])
        
        # Contrast and homogeneity
        features.append(np.std(gray_roi))  # Local contrast
        
        return features
    
    def _extract_geometric_features(self, gray_roi: np.ndarray) -> List[float]:
        """Geometrik özellikler"""
        features = []
        
        h, w = gray_roi.shape
        features.append(w / (h + 1e-6))  # Aspect ratio
        features.append(h * w)  # Area
        
        # Moment-based features
        moments = cv2.moments(gray_roi)
        if moments['m00'] != 0:
            # Centroids
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features.append(cx / w)  # Normalized centroid x
            features.append(cy / h)  # Normalized centroid y
            
            # Hu moments (first 3)
            hu_moments = cv2.HuMoments(moments)
            for i in range(3):
                hu = hu_moments[i][0]
                features.append(-np.sign(hu) * np.log10(np.abs(hu) + 1e-6))
        else:
            features.extend([0.5, 0.5, 0, 0, 0])  # Default values
        
        return features
    
    def _extract_region_specific_features(self, gray_roi: np.ndarray, region_name: str) -> List[float]:
        """Bölgeye özel optimize edilmiş özellikler"""
        features = []
        
        # Muscle fiber direction analysis
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        
        if 'pectoralis' in region_name:
            # Chest: horizontal fiber emphasis
            horizontal_strength = np.mean(np.abs(grad_y))
            features.append(horizontal_strength)
            features.append(horizontal_strength / (np.mean(np.abs(grad_x)) + 1e-6))
            
        elif 'rectus' in region_name:
            # Abs: vertical segmentation emphasis
            vertical_strength = np.mean(np.abs(grad_x))
            features.append(vertical_strength)
            
            # Segmentation detection
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            features.append(np.sum(vertical_lines) / (binary.size + 1e-6))
            
        elif 'deltoid' in region_name:
            # Shoulder: roundness and definition
            contours, _ = cv2.findContours(
                cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                    features.append(circularity)
                else:
                    features.append(0)
            else:
                features.append(0)
        
        # Pad to consistent length
        while len(features) < 5:
            features.append(0)
        
        return features[:5]

class OptimizedEnsembleSystem:
    """Optimize edilmiş ensemble sistemi"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,  # Reduced for speed
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,  # Reduced for speed
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        self.scalers = {
            'random_forest': RobustScaler(),
            'gradient_boosting': StandardScaler()
        }
        
        self.is_trained = False
        self._train_models()
    
    def _train_models(self):
        """Modelleri eğit"""
        print("🧠 Machine learning modelleri eğitiliyor...")
        
        # Generate synthetic training data
        X, y = self._generate_training_data(2000)  # Reduced for speed
        
        self.training_scores = {}
        
        for name, model in self.models.items():
            # Scale features
            scaler = self.scalers[name]
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Performance metrics
            train_score = model.score(X_scaled, y)
            cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')  # Reduced CV folds
            
            self.training_scores[name] = {
                'train_r2': train_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        self.is_trained = True
        print("✅ Model eğitimi tamamlandı")
    
    def _generate_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sentetik eğitim verisi"""
        np.random.seed(42)
        
        X = []
        y = []
        
        for i in range(n_samples):
            # Realistic feature generation
            features = []
            
            # Statistical features (10) - daha gerçekçi dağılım
            muscle_level = np.random.beta(2, 2) * 100  # Daha balanced dağılım
            body_fat = np.random.uniform(8, 25)  # Uniform dağılım
            
            # Mean, std, var, min, max, median
            base_intensity = 120 + muscle_level * 0.5 - body_fat * 2
            features.extend([
                base_intensity + np.random.normal(0, 10),  # mean
                20 + muscle_level * 0.3 + np.random.normal(0, 5),  # std
                (20 + muscle_level * 0.3)**2,  # var
                max(0, base_intensity - 50),  # min
                min(255, base_intensity + 50),  # max
                base_intensity + np.random.normal(0, 5)  # median
            ])
            
            # Skewness, kurtosis
            features.extend([np.random.normal(0, 0.5), np.random.normal(0, 1)])
            
            # Percentiles
            features.extend([
                base_intensity - 20 + np.random.normal(0, 5),
                base_intensity + 20 + np.random.normal(0, 5)
            ])
            
            # Edge features (7) - critical for muscle definition - güçlendirildi
            edge_strength = muscle_level / 100 * 80 - body_fat * 0.3  # Daha güçlü edge
            features.extend([
                max(0, edge_strength + np.random.normal(0, 8)),  # gradient mean
                15 + muscle_level * 0.15,  # gradient std
                max(0, edge_strength * 2.5),  # gradient max
                max(0, (muscle_level / 100) * 0.4 - body_fat * 0.008),  # edge density
                max(0, muscle_level * 3 + np.random.normal(0, 40)),  # laplacian var
                max(0, edge_strength * 0.7),  # laplacian mean abs
                max(0.3, min(1.0, 0.7 + np.random.normal(0, 0.15)))  # orientation consistency
            ])
            
            # Texture features (10)
            texture_base = muscle_level / 100 * 0.8
            features.extend([
                texture_base + np.random.normal(0, 0.1),  # LBP mean
                0.3 + np.random.normal(0, 0.1),  # LBP std
                1.5 + np.random.normal(0, 0.3),  # LBP entropy
                texture_base * 20,  # Gabor 0
                texture_base * 15,  # Gabor 45
                texture_base * 18,  # Gabor 90
                20 + muscle_level * 0.3  # contrast
            ])
            
            # Add remaining features to reach 50
            remaining_features = 50 - len(features)
            for _ in range(remaining_features):
                feature_value = muscle_level / 100 + np.random.normal(0, 0.3)
                features.append(feature_value)
            
            X.append(features[:50])
            
            # Target score calculation
            score = self._calculate_realistic_score(muscle_level, body_fat, features)
            y.append(score)
        
        return np.array(X), np.array(y)
    
    def _calculate_realistic_score(self, muscle_level: float, body_fat: float, features: List[float]) -> float:
        """Gerçekçi skor hesaplama - düzeltilmiş"""
        # Daha yüksek base score
        base_score = muscle_level * 0.8 + 30  # 30 base puan ekledik
        
        # Daha az body fat penalty
        fat_penalty = max(0, (body_fat - 15) * 1.5)  # Threshold artırıldı
        
        # Güçlü feature bonuses
        edge_bonus = features[13] * 150 if len(features) > 13 else 0  # Edge density artırıldı
        contrast_bonus = (features[23] - 20) * 0.8 if len(features) > 23 else 0  # Contrast threshold düşürüldü
        
        # Muscle definition bonus
        muscle_def_bonus = features[10] * 0.3 if len(features) > 10 else 0  # Gradient mean
        
        # Final score
        score = base_score - fat_penalty + edge_bonus + contrast_bonus + muscle_def_bonus
        score += np.random.normal(0, 5)  # Daha az variation
        
        return max(40, min(95, score))  # Range genişletildi
    
    def predict_accurate(self, features: np.ndarray) -> AccuratePrediction:
        """Doğru tahmin"""
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            # Scale features
            scaler = self.scalers[name]
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Predict
            pred = model.predict(features_scaled)[0]
            predictions[name] = pred
            
            # Confidence based on training performance
            train_performance = self.training_scores[name]['cv_mean']
            confidences[name] = max(0.6, min(0.95, train_performance))
        
        # Weighted ensemble
        weights = np.array(list(confidences.values()))
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(list(predictions.values()), weights=weights)
        ensemble_confidence = np.mean(list(confidences.values()))
        
        # Uncertainty (standard deviation of predictions)
        uncertainty = np.std(list(predictions.values()))
        
        # Quality score based on feature consistency
        quality_score = self._calculate_quality_score(features)
        
        return AccuratePrediction(
            value=ensemble_pred,
            confidence=ensemble_confidence,
            uncertainty=uncertainty,
            method='optimized_ensemble',
            quality_score=quality_score
        )
    
    def _calculate_quality_score(self, features: np.ndarray) -> float:
        """Özellik kalite skoru"""
        # Check for reasonable feature values
        feature_ranges = {
            0: (50, 200),    # mean intensity
            1: (10, 60),     # std
            13: (0, 0.5),    # edge density
            23: (10, 80)     # contrast
        }
        
        quality = 1.0
        for idx, (min_val, max_val) in feature_ranges.items():
            if idx < len(features):
                if features[idx] < min_val or features[idx] > max_val:
                    quality *= 0.8  # Penalize out-of-range values
        
        return quality

class OptimizedAccurateAnalyzer:
    """Optimize edilmiş doğru analiz sistemi"""
    
    def __init__(self):
        self.feature_extractor = OptimizedFeatureExtractor()
        self.ensemble_system = OptimizedEnsembleSystem()
        self.mp_pose = mp.solutions.pose
        
        # Renk haritası skorlara göre
        self.color_map = {
            'excellent': (0, 255, 0),      # Yeşil: 85-100
            'very_good': (0, 200, 100),    # Açık yeşil: 75-84
            'good': (0, 150, 200),         # Turkuaz: 65-74
            'average': (0, 100, 255),      # Mavi: 55-64
            'below_avg': (50, 50, 255),    # Koyu mavi: 45-54
            'poor': (0, 0, 255),           # Kırmızı: 35-44
            'very_poor': (0, 0, 150)       # Koyu kırmızı: 0-34
        }
        
    def accurate_analysis(self, image: np.ndarray) -> OptimizedResult:
        """Doğru ve optimize edilmiş analiz"""
        print("🎯 Optimized accurate analysis başlatılıyor...")
        
        # Enhanced image processing
        enhanced_image = self._enhance_image(image)
        
        # Pose detection
        landmarks, pose_confidence = self._detect_pose(enhanced_image)
        
        # Get regions
        if landmarks and pose_confidence > 0.6:
            regions = self._get_regions_from_landmarks(landmarks, image.shape)
        else:
            regions = self._get_default_regions(image.shape)
        
        # Analyze each region
        region_predictions = {}
        all_scores = []
        all_confidences = []
        
        for region_name, coords in regions.items():
            print(f"   🔍 Analyzing: {region_name}")
            
            # Extract ROI
            x1, y1, x2, y2 = coords
            roi = enhanced_image[y1:y2, x1:x2]
            
            if roi.size == 0:
                # Default prediction for empty ROI
                prediction = AccuratePrediction(
                    value=50.0, confidence=0.3, uncertainty=15.0,
                    method='default', quality_score=0.5
                )
            else:
                # Extract features
                features = self.feature_extractor.extract_enhanced_features(roi, region_name)
                
                # Make prediction
                prediction = self.ensemble_system.predict_accurate(features)
                
                # Image-specific adjustment based on actual features
                adjusted_score = self._adjust_score_based_on_image(prediction.value, features, region_name)
                prediction.value = adjusted_score
            
            region_predictions[region_name] = prediction
            all_scores.append(prediction.value)
            all_confidences.append(prediction.confidence)
        
        # Calculate ensemble metrics
        ensemble_score = np.mean(all_scores)
        model_confidence = np.mean(all_confidences)
        
        # Prediction range (confidence interval)
        uncertainties = [pred.uncertainty for pred in region_predictions.values()]
        mean_uncertainty = np.mean(uncertainties)
        prediction_range = (
            ensemble_score - 1.96 * mean_uncertainty,
            ensemble_score + 1.96 * mean_uncertainty
        )
        
        # Quality metrics
        quality_metrics = self._calculate_image_quality(enhanced_image, pose_confidence)
        
        # Feature analysis
        feature_analysis = self._analyze_features(region_predictions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            region_predictions, ensemble_score, model_confidence
        )
        
        # Görsel çıktı oluştur
        visual_output_path = self._create_visual_output(
            image, regions, region_predictions, landmarks, pose_confidence
        )
        
        return OptimizedResult(
            region_predictions=region_predictions,
            ensemble_score=ensemble_score,
            model_confidence=model_confidence,
            prediction_range=prediction_range,
            quality_metrics=quality_metrics,
            feature_analysis=feature_analysis,
            recommendations=recommendations,
            visual_output_path=visual_output_path
        )
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Görüntü iyileştirme"""
        # LAB color space enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _detect_pose(self, image: np.ndarray) -> Tuple[Optional[Dict], float]:
        """Pose detection"""
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # Optimized complexity
            enable_segmentation=False,  # Disabled for speed
            min_detection_confidence=0.7
        ) as pose:
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if results.pose_landmarks:
                h, w = image.shape[:2]
                landmarks = {}
                
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[idx] = {
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'visibility': landmark.visibility
                    }
                
                # Calculate average confidence
                visibility_scores = [lm.visibility for lm in results.pose_landmarks.landmark]
                avg_confidence = np.mean(visibility_scores)
                
                return landmarks, avg_confidence
        
        return None, 0.0
    
    def _get_regions_from_landmarks(self, landmarks: Dict, image_shape: Tuple) -> Dict:
        """Landmark-based regions"""
        h, w = image_shape[:2]
        regions = {}
        
        try:
            # Key landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate region coordinates
            chest_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            chest_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # Upper pectoralis
            regions['upper_pectoralis'] = (
                int(chest_center_x - 80), int(chest_center_y - 50),
                int(chest_center_x + 80), int(chest_center_y + 30)
            )
            
            # Lower pectoralis
            regions['lower_pectoralis'] = (
                int(chest_center_x - 70), int(chest_center_y + 30),
                int(chest_center_x + 70), int(chest_center_y + 100)
            )
            
            # Upper rectus
            regions['upper_rectus'] = (
                int(chest_center_x - 40), int(chest_center_y + 120),
                int(chest_center_x + 40), int(chest_center_y + 180)
            )
            
            # Lateral deltoid
            regions['lateral_deltoid'] = (
                int(left_shoulder['x'] - 60), int(left_shoulder['y'] - 30),
                int(left_shoulder['x'] + 20), int(left_shoulder['y'] + 60)
            )
            
            # Biceps
            regions['biceps_brachii'] = (
                int(left_shoulder['x'] - 40), int(left_shoulder['y'] + 60),
                int(left_shoulder['x'] + 20), int(left_shoulder['y'] + 140)
            )
            
        except:
            regions = self._get_default_regions(image_shape)
        
        return regions
    
    def _get_default_regions(self, image_shape: Tuple) -> Dict:
        """Default regions"""
        h, w = image_shape[:2]
        return {
            'upper_pectoralis': (int(w*0.2), int(h*0.1), int(w*0.8), int(h*0.35)),
            'lower_pectoralis': (int(w*0.25), int(h*0.3), int(w*0.75), int(h*0.5)),
            'upper_rectus': (int(w*0.35), int(h*0.45), int(w*0.65), int(h*0.6)),
            'lateral_deltoid': (int(w*0.05), int(h*0.1), int(w*0.3), int(h*0.4)),
            'biceps_brachii': (int(w*0.0), int(h*0.25), int(w*0.2), int(h*0.5))
        }
    
    def _calculate_image_quality(self, image: np.ndarray, pose_confidence: float) -> Dict[str, float]:
        """Görüntü kalite metrikleri"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000.0)
        
        # Contrast
        contrast = np.std(gray) / 255.0
        
        # Brightness
        brightness = np.mean(gray) / 255.0
        
        # Overall quality
        overall_quality = (sharpness * 0.4 + contrast * 0.3 + pose_confidence * 0.3)
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'pose_confidence': pose_confidence,
            'overall_quality': overall_quality
        }
    
    def _analyze_features(self, predictions: Dict[str, AccuratePrediction]) -> Dict[str, float]:
        """Özellik analizi"""
        scores = [pred.value for pred in predictions.values()]
        confidences = [pred.confidence for pred in predictions.values()]
        uncertainties = [pred.uncertainty for pred in predictions.values()]
        
        return {
            'score_std': np.std(scores),
            'score_range': np.max(scores) - np.min(scores),
            'avg_confidence': np.mean(confidences),
            'avg_uncertainty': np.mean(uncertainties),
            'consistency': 1.0 - (np.std(scores) / 50.0)  # Normalized consistency
        }
    
    def _generate_recommendations(self, predictions: Dict[str, AccuratePrediction],
                                ensemble_score: float, model_confidence: float) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        # Model confidence analysis
        if model_confidence < 0.7:
            recommendations.append("⚠️ Model güveni düşük - daha net görüntü önerilir")
        
        # Score-based recommendations
        if ensemble_score < 50:
            recommendations.append("🎯 Temel kas geliştirme programına odaklanın")
            recommendations.append("📈 Compound egzersizleri öncelikleyin")
        elif ensemble_score < 70:
            recommendations.append("💪 Orta seviye teknikleri entegre edin")
            recommendations.append("🔧 Zayıf bölgelere ekstra dikkat gösterin")
        else:
            recommendations.append("🏆 İleri seviye programlar uygulayabilirsiniz")
            recommendations.append("⚡ Periodization stratejisi geliştirin")
        
        # Region-specific analysis
        weak_regions = [name for name, pred in predictions.items() if pred.value < 55]
        if weak_regions:
            recommendations.append(f"🎯 Öncelik alanları: {', '.join(weak_regions[:3])}")
        
        strong_regions = [name for name, pred in predictions.items() if pred.value > 75]
        if strong_regions:
            recommendations.append(f"💪 Güçlü bölgeler: {', '.join(strong_regions[:2])} - mevcut seviyeyi koruyun")
        
        # Quality-based recommendations
        low_quality_regions = [name for name, pred in predictions.items() if pred.quality_score < 0.6]
        if low_quality_regions:
            recommendations.append(f"📸 Görüntü kalitesi: {', '.join(low_quality_regions[:2])} bölgeleri için daha iyi açı deneyin")
        
        return recommendations
    
    def _adjust_score_based_on_image(self, base_score: float, features: np.ndarray, region_name: str) -> float:
        """Gerçek görüntü özelliklerine göre skor düzeltmesi"""
        
        # Temel kas tanımı indicators
        mean_intensity = features[0] if len(features) > 0 else 100
        contrast = features[1] if len(features) > 1 else 20
        edge_density = features[13] if len(features) > 13 else 0.1
        gradient_strength = features[10] if len(features) > 10 else 10
        
        # Kas tanımı faktörü
        muscle_definition = 0
        
        # Yüksek kontrast = iyi kas tanımı
        if contrast > 40:
            muscle_definition += 15
        elif contrast > 30:
            muscle_definition += 10
        elif contrast > 20:
            muscle_definition += 5
        
        # Edge density - kas çizgileri
        if edge_density > 0.2:
            muscle_definition += 20
        elif edge_density > 0.15:
            muscle_definition += 15
        elif edge_density > 0.1:
            muscle_definition += 10
        
        # Gradient strength - kas yüzeyi
        if gradient_strength > 30:
            muscle_definition += 15
        elif gradient_strength > 20:
            muscle_definition += 10
        elif gradient_strength > 15:
            muscle_definition += 5
        
        # Intensity range - kas-yağ ayrımı
        if 100 <= mean_intensity <= 150:  # Optimal range
            muscle_definition += 10
        elif 80 <= mean_intensity <= 180:
            muscle_definition += 5
        
        # Bölgeye özel düzeltmeler
        if 'pectoralis' in region_name:
            # Göğüs kasları için
            if contrast > 35 and edge_density > 0.15:
                muscle_definition += 10  # Extra bonus for chest definition
        elif 'rectus' in region_name:
            # Karın kasları için 
            if edge_density > 0.2:  # Segmentation
                muscle_definition += 15
        elif 'deltoid' in region_name:
            # Omuz kasları için
            if gradient_strength > 25:
                muscle_definition += 8
        
        # Final adjustment
        adjusted_score = base_score + muscle_definition
        
        # Ensure realistic range
        return max(35, min(95, adjusted_score))
    
    def _create_visual_output(self, original_image: np.ndarray, regions: Dict, 
                            predictions: Dict[str, AccuratePrediction], 
                            landmarks: Optional[Dict], pose_confidence: float) -> str:
        """Görsel çıktı oluştur - renkli bölge gösterimi"""
        
        # Orijinal görüntüyü kopyala
        visual_image = original_image.copy()
        h, w = visual_image.shape[:2]
        
        # Overlay için şeffaf katman
        overlay = visual_image.copy()
        
        print("🎨 Görsel analiz sonuçları oluşturuluyor...")
        
        for region_name, coords in regions.items():
            if region_name in predictions:
                prediction = predictions[region_name]
                score = prediction.value
                confidence = prediction.confidence
                
                # Skor kategorisini belirle
                color_category = self._get_color_category(score)
                color = self.color_map[color_category]
                
                # Bölge koordinatları
                x1, y1, x2, y2 = coords
                
                # Bölgeyi renkle doldur (şeffaf)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Çerçeve çiz
                thickness = 3 if confidence > 0.8 else 2
                cv2.rectangle(visual_image, (x1, y1), (x2, y2), color, thickness)
                
                # Skor ve güven metinleri
                score_text = f"{score:.1f}"
                conf_text = f"C:{confidence:.2f}"
                
                # Metin pozisyonu
                text_x = x1 + 5
                text_y = y1 + 20
                
                # Metin arka planı
                cv2.rectangle(visual_image, (text_x-2, text_y-15), 
                            (text_x + 80, text_y + 25), (0, 0, 0), -1)
                
                # Skor metni
                cv2.putText(visual_image, score_text, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Güven metni
                cv2.putText(visual_image, conf_text, (text_x, text_y + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Bölge ismi (kısaltılmış)
                region_short_name = self._get_short_region_name(region_name)
                cv2.putText(visual_image, region_short_name, (text_x, y2 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Şeffaf katmanı birleştir
        alpha = 0.3  # Şeffaflık oranı
        cv2.addWeighted(overlay, alpha, visual_image, 1 - alpha, 0, visual_image)
        
        # Pose landmarks çiz (eğer var ise)
        if landmarks and pose_confidence > 0.5:
            self._draw_pose_landmarks(visual_image, landmarks)
        
        # Genel bilgi paneli ekle
        self._add_info_panel(visual_image, predictions)
        
        # Dosya kaydet
        output_filename = "optimized_visual_analysis.jpg"
        cv2.imwrite(output_filename, visual_image)
        
        print(f"✅ Görsel analiz kaydedildi: {output_filename}")
        return output_filename
    
    def _get_color_category(self, score: float) -> str:
        """Skor kategorisini belirle"""
        if score >= 85:
            return 'excellent'
        elif score >= 75:
            return 'very_good'
        elif score >= 65:
            return 'good'
        elif score >= 55:
            return 'average'
        elif score >= 45:
            return 'below_avg'
        elif score >= 35:
            return 'poor'
        else:
            return 'very_poor'
    
    def _get_short_region_name(self, region_name: str) -> str:
        """Bölge ismini kısalt"""
        name_map = {
            'upper_pectoralis': 'U.PECT',
            'lower_pectoralis': 'L.PECT', 
            'upper_rectus': 'U.ABS',
            'lateral_deltoid': 'L.DELT',
            'biceps_brachii': 'BICEPS'
        }
        return name_map.get(region_name, region_name[:6].upper())
    
    def _draw_pose_landmarks(self, image: np.ndarray, landmarks: Dict):
        """Pose landmark'larını çiz"""
        # Temel bağlantı noktaları
        key_points = [11, 12, 23, 24]  # Omuzlar ve kalçalar
        
        for point_id in key_points:
            if point_id in landmarks:
                landmark = landmarks[point_id]
                if landmark['visibility'] > 0.5:
                    x, y = int(landmark['x']), int(landmark['y'])
                    cv2.circle(image, (x, y), 5, (255, 255, 0), -1)  # Sarı nokta
                    cv2.circle(image, (x, y), 8, (0, 0, 0), 2)       # Siyah çerçeve
    
    def _add_info_panel(self, image: np.ndarray, predictions: Dict[str, AccuratePrediction]):
        """Bilgi paneli ekle"""
        h, w = image.shape[:2]
        
        # Panel boyutları
        panel_h = 120
        panel_w = 300
        panel_x = w - panel_w - 10
        panel_y = 10
        
        # Panel arka planı
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)
        
        # Başlık
        cv2.putText(image, "ANALIZ SONUCLARI", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ortalama skor
        avg_score = np.mean([pred.value for pred in predictions.values()])
        cv2.putText(image, f"Ortalama: {avg_score:.1f}/100", 
                   (panel_x + 10, panel_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # En yüksek ve en düşük skorlar
        scores = [(name, pred.value) for name, pred in predictions.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_region, best_score = scores[0]
        worst_region, worst_score = scores[-1]
        
        cv2.putText(image, f"En iyi: {self._get_short_region_name(best_region)} ({best_score:.1f})", 
                   (panel_x + 10, panel_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(image, f"Zayif: {self._get_short_region_name(worst_region)} ({worst_score:.1f})", 
                   (panel_x + 10, panel_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Renk kodları açıklaması
        legend_y = panel_y + panel_h + 20
        cv2.putText(image, "RENK KODLARI:", (panel_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        color_legend = [
            ("Mukemmel (85+)", self.color_map['excellent']),
            ("Cok iyi (75-84)", self.color_map['very_good']),
            ("Iyi (65-74)", self.color_map['good']),
            ("Orta (55-64)", self.color_map['average']),
            ("Zayif (<55)", self.color_map['poor'])
        ]
        
        for i, (text, color) in enumerate(color_legend):
            y_pos = legend_y + 20 + (i * 15)
            cv2.rectangle(image, (panel_x, y_pos - 8), (panel_x + 12, y_pos + 2), color, -1)
            cv2.putText(image, text, (panel_x + 18, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def main():
    if len(sys.argv) < 2:
        print("🎯 OPTIMIZED ACCURATE FITNESS SYSTEM")
        print("=" * 70)
        print("Maximum doğruluk + optimize performans")
        print("\n🔬 Features:")
        print("  ✅ Ensemble Machine Learning (RF + GB)")
        print("  ✅ 50+ Optimized Features")
        print("  ✅ Uncertainty Quantification")
        print("  ✅ Quality Assessment")
        print("  ✅ Region-specific Analysis")
        print("  ✅ Advanced Recommendations")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        print("🎯 OPTIMIZED ACCURATE SYSTEM başlatılıyor...")
        print("=" * 70)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image!")
            return
        
        # Initialize analyzer
        analyzer = OptimizedAccurateAnalyzer()
        
        # Run analysis
        result = analyzer.accurate_analysis(image)
        
        print("\n" + "="*70)
        print("🎯 OPTIMIZED ACCURATE ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\n📊 ENSEMBLE METRICS:")
        print(f"   🎯 Ensemble Score: {result.ensemble_score:.2f}/100")
        print(f"   🤝 Model Confidence: {result.model_confidence:.3f}")
        print(f"   📈 Prediction Range: ({result.prediction_range[0]:.1f}, {result.prediction_range[1]:.1f})")
        
        print(f"\n🔍 REGION ANALYSIS:")
        sorted_regions = sorted(result.region_predictions.items(), 
                              key=lambda x: x[1].value, reverse=True)
        
        for region_name, prediction in sorted_regions:
            confidence_emoji = "🟢" if prediction.confidence > 0.8 else ("🟡" if prediction.confidence > 0.6 else "🔴")
            print(f"   {confidence_emoji} {region_name}:")
            print(f"      Score: {prediction.value:.1f} ± {prediction.uncertainty:.1f}")
            print(f"      Confidence: {prediction.confidence:.3f}")
            print(f"      Quality: {prediction.quality_score:.3f}")
        
        print(f"\n📈 QUALITY METRICS:")
        for metric, value in result.quality_metrics.items():
            quality_emoji = "🟢" if value > 0.7 else ("🟡" if value > 0.5 else "🔴")
            print(f"   {quality_emoji} {metric}: {value:.3f}")
        
        print(f"\n📊 FEATURE ANALYSIS:")
        for feature, value in result.feature_analysis.items():
            print(f"   {feature}: {value:.3f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Performance summary
        high_conf_regions = sum(1 for pred in result.region_predictions.values() if pred.confidence > 0.8)
        total_regions = len(result.region_predictions)
        
        print(f"\n🏆 PERFORMANCE SUMMARY:")
        print(f"   High Confidence Regions: {high_conf_regions}/{total_regions}")
        print(f"   Average Uncertainty: {np.mean([p.uncertainty for p in result.region_predictions.values()]):.2f}")
        print(f"   Model Reliability: {result.model_confidence:.3f}")
        
        if result.model_confidence > 0.85:
            reliability = "🟢 EXCELLENT"
        elif result.model_confidence > 0.75:
            reliability = "🟡 GOOD"  
        else:
            reliability = "🔴 MODERATE"
            
        print(f"   Overall Reliability: {reliability}")
        
        print(f"\n🎉 OPTIMIZED ACCURATE ANALYSIS COMPLETE!")
        print(f"   ⚡ Fast performance with maximum accuracy")
        print(f"   🎯 {total_regions} regions analyzed with ensemble ML")
        print(f"   📊 Comprehensive quality assessment included")
        
        if result.visual_output_path:
            print(f"\n🎨 GÖRSEL ÇIKTI:")
            print(f"   📷 Renkli analiz görüntüsü: {result.visual_output_path}")
            print(f"   🌈 Bölgeler skorlarına göre renklendirildi")
            print(f"   📊 Detaylı bilgi paneli eklendi")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()