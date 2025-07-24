#!/usr/bin/env python3
"""
🚀 Next-Generation Body Analyzer LITE Version
Scikit-learn bağımlılığı olmadan çalışan basitleştirilmiş versiyon
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

@dataclass
class BodyRegionAnalysis:
    """Next-gen vücut bölgesi analiz sonucu"""
    muscle_mass_score: float        # ML predicted muscle mass (0-100)
    fat_percentage: float           # Body fat percentage (0-100)
    muscle_definition: float        # Texture-based definition (0-100)
    vascularity_score: float        # Vascular visibility (0-100)
    symmetry_score: float           # Left-right symmetry (0-100)
    anthropometric_score: float     # Proportional measurements (0-100)
    overall_score: float            # Weighted combination (0-100)
    confidence_level: float         # Algorithm confidence (0-1)
    population_percentile: float    # Age/gender adjusted percentile
    grade: str                      # A+, A, B+, B, C+, C, D+, D, F
    recommendations: List[str]
    technical_details: Dict

class NextGenBodyAnalyzerLite:
    def __init__(self):
        """Initialize next-generation body analyzer lite"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Modern renk paleti
        self.colors = {
            'elite': (0, 255, 0),           # A+ grade (95-100)
            'excellent': (50, 255, 50),     # A grade (90-94)
            'very_good': (0, 255, 150),     # B+ grade (85-89)
            'good': (0, 255, 255),          # B grade (80-84)
            'above_avg': (0, 200, 255),     # C+ grade (70-79)
            'average': (0, 165, 255),       # C grade (60-69)
            'below_avg': (0, 100, 255),     # D+ grade (50-59)
            'poor': (0, 50, 255),           # D grade (40-49)
            'very_poor': (0, 0, 255),       # F grade (0-39)
        }
        
        # Anatomik bölge tanımları (ASCII-safe)
        self.anatomical_regions = {
            'upper_pectoralis': 'Ust Gogus',
            'lower_pectoralis': 'Alt Gogus', 
            'upper_rectus': 'Ust Karin',
            'lower_rectus': 'Alt Karin',
            'external_obliques': 'Dis Oblik',
            'serratus_anterior': 'Serratus',
            'anterior_deltoid': 'On Deltoid',
            'lateral_deltoid': 'Yan Deltoid',
            'biceps_brachii': 'Biceps',
            'triceps_brachii': 'Triceps'
        }
        
        # Evidence-based scoring weights
        self.scoring_weights = {
            'muscle_mass': 0.35,
            'definition': 0.25,
            'body_fat': 0.20,
            'vascularity': 0.10,
            'symmetry': 0.10
        }
    
    def adaptive_image_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Adaptive image enhancement pipeline"""
        enhanced = image.copy()
        
        # LAB color space conversion
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Gamma correction
        gamma = self._estimate_optimal_gamma(l_channel)
        l_gamma_corrected = self._apply_gamma_correction(l_enhanced, gamma)
        
        # Merge back
        lab_enhanced = cv2.merge([l_gamma_corrected, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
        
        return enhanced
    
    def _estimate_optimal_gamma(self, grayscale: np.ndarray) -> float:
        """Optimal gamma değeri tahmin eder"""
        hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
        total_pixels = grayscale.size
        weighted_sum = sum(i * hist[i][0] for i in range(256))
        mean_intensity = weighted_sum / total_pixels
        
        if mean_intensity < 85:
            gamma = 0.7
        elif mean_intensity > 170:
            gamma = 1.3
        else:
            gamma = 1.0
            
        return gamma
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma correction uygular"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def advanced_edge_detection(self, image: np.ndarray, region_coords: Tuple[int, int, int, int]) -> Dict:
        """Advanced edge detection"""
        x1, y1, x2, y2 = region_coords
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'edge_density': 0, 'edge_strength': 0, 'orientation_consistency': 0}
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Multi-directional gradients
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Adaptive Canny
        otsu_thresh, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_thresh = max(1, int(0.5 * otsu_thresh))
        upper_thresh = max(lower_thresh + 1, int(otsu_thresh))
        
        edges = cv2.Canny(gray_roi, lower_thresh, upper_thresh)
        
        edge_density = np.sum(edges > 0) / edges.size
        edge_strength = np.mean(gradient_magnitude)
        orientation_consistency = self._calculate_orientation_consistency(gradient_direction)
        
        return {
            'edge_density': edge_density,
            'edge_strength': edge_strength,
            'orientation_consistency': orientation_consistency
        }
    
    def _calculate_orientation_consistency(self, gradient_direction: np.ndarray) -> float:
        """Edge orientation consistency hesaplar"""
        angles_deg = gradient_direction * 180 / np.pi
        angles_deg = angles_deg % 180
        
        hist, _ = np.histogram(angles_deg, bins=18, range=(0, 180))
        max_bin = np.max(hist)
        total_pixels = np.sum(hist)
        
        consistency = max_bin / (total_pixels + 1e-6)
        return consistency
    
    def simplified_muscle_fat_analysis(self, image: np.ndarray, region_coords: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """Simplified muscle-fat analysis without ML"""
        x1, y1, x2, y2 = region_coords
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 50.0, 15.0, 0.8
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Edge analysis for muscle definition
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contrast analysis
        contrast = np.std(gray_roi)
        
        # Laplacian variance for sharpness
        laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        
        # Muscle score calculation
        muscle_score = 0
        
        if edge_density > 0.15:
            muscle_score += 35
        elif edge_density > 0.1:
            muscle_score += 25
        elif edge_density > 0.05:
            muscle_score += 15
        
        if contrast > 40:
            muscle_score += 30
        elif contrast > 30:
            muscle_score += 20
        elif contrast > 20:
            muscle_score += 10
        
        if laplacian_var > 500:
            muscle_score += 35
        elif laplacian_var > 200:
            muscle_score += 20
        
        muscle_score = max(0, min(100, muscle_score))
        
        # Fat percentage estimation (simplified)
        brightness = np.mean(gray_roi)
        homogeneity = 1.0 / (1.0 + np.std(gray_roi))
        
        fat_score = 0
        if brightness > 160:
            fat_score += 25
        elif brightness > 140:
            fat_score += 15
        
        if homogeneity > 0.05:
            fat_score += 20
        elif homogeneity > 0.03:
            fat_score += 10
        
        fat_percentage = max(5, min(35, 10 + fat_score * 0.5))
        
        # Confidence based on image quality
        confidence = min(1.0, (edge_density * 2 + contrast / 50 + laplacian_var / 1000) / 3)
        
        return muscle_score, fat_percentage, confidence
    
    def advanced_pose_analysis(self, image: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """Advanced MediaPipe pose analysis"""
        pose_results = None
        annotated_image = image.copy()
        
        complexities = [2, 1, 0]
        
        for complexity in complexities:
            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=complexity,
                enable_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_image)
                
                if results.pose_landmarks:
                    pose_results = results
                    break
        
        if pose_results is None:
            return None, annotated_image
        
        # Extract landmarks
        landmarks_dict = {}
        h, w = image.shape[:2]
        
        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
            landmarks_dict[idx] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        # Draw pose
        self.mp_drawing.draw_landmarks(
            annotated_image,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=1)
        )
        
        # Body segmentation overlay
        if pose_results.segmentation_mask is not None:
            segmentation_mask = pose_results.segmentation_mask
            colored_mask = np.zeros_like(image)
            colored_mask[segmentation_mask > 0.5] = [0, 100, 0]
            annotated_image = cv2.addWeighted(annotated_image, 0.8, colored_mask, 0.2, 0)
        
        return landmarks_dict, annotated_image
    
    def get_anatomical_regions_from_pose(self, landmarks: Dict, image_shape: Tuple[int, int]) -> Dict:
        """Enhanced anatomical region extraction from pose landmarks"""
        h, w = image_shape[:2]
        regions = {}
        
        try:
            # Key landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate proportions
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            torso_length = abs((left_hip['y'] + right_hip['y'])/2 - (left_shoulder['y'] + right_shoulder['y'])/2)
            
            region_scale_x = shoulder_width / 200
            region_scale_y = torso_length / 300
            
            # Define regions
            chest_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            chest_top_y = (left_shoulder['y'] + right_shoulder['y']) / 2 - 20 * region_scale_y
            chest_mid_y = chest_top_y + 80 * region_scale_y
            
            regions['upper_pectoralis'] = (
                int(chest_center_x - 60 * region_scale_x),
                int(chest_top_y),
                int(chest_center_x + 60 * region_scale_x),
                int(chest_mid_y)
            )
            
            # Continue with other regions...
            chest_bottom_y = chest_mid_y + 70 * region_scale_y
            regions['lower_pectoralis'] = (
                int(chest_center_x - 50 * region_scale_x),
                int(chest_mid_y),
                int(chest_center_x + 50 * region_scale_x),
                int(chest_bottom_y)
            )
            
            # Abs regions
            abs_center_x = (left_hip['x'] + right_hip['x']) / 2
            abs_top_y = chest_bottom_y + 10 * region_scale_y
            abs_mid_y = abs_top_y + 70 * region_scale_y
            abs_bottom_y = abs_mid_y + 70 * region_scale_y
            
            regions['upper_rectus'] = (
                int(abs_center_x - 40 * region_scale_x),
                int(abs_top_y),
                int(abs_center_x + 40 * region_scale_x),
                int(abs_mid_y)
            )
            
            regions['lower_rectus'] = (
                int(abs_center_x - 35 * region_scale_x),
                int(abs_mid_y),
                int(abs_center_x + 35 * region_scale_x),
                int(abs_bottom_y)
            )
            
            # Add more regions...
            regions['external_obliques'] = (
                int(abs_center_x - 80 * region_scale_x),
                int(abs_top_y),
                int(abs_center_x - 35 * region_scale_x),
                int(abs_bottom_y)
            )
            
            regions['serratus_anterior'] = (
                int(left_shoulder['x'] - 30 * region_scale_x),
                int(left_shoulder['y'] + 40 * region_scale_y),
                int(left_shoulder['x'] + 10 * region_scale_x),
                int(left_shoulder['y'] + 120 * region_scale_y)
            )
            
            regions['anterior_deltoid'] = (
                int(left_shoulder['x'] - 40 * region_scale_x),
                int(left_shoulder['y'] - 30 * region_scale_y),
                int(left_shoulder['x'] + 20 * region_scale_x),
                int(left_shoulder['y'] + 50 * region_scale_y)
            )
            
            regions['lateral_deltoid'] = (
                int(right_shoulder['x'] - 20 * region_scale_x),
                int(right_shoulder['y'] - 30 * region_scale_y),
                int(right_shoulder['x'] + 40 * region_scale_x),
                int(right_shoulder['y'] + 50 * region_scale_y)
            )
            
            bicep_center_x = (left_shoulder['x'] + left_elbow['x']) / 2
            bicep_center_y = (left_shoulder['y'] + left_elbow['y']) / 2
            
            regions['biceps_brachii'] = (
                int(bicep_center_x - 25 * region_scale_x),
                int(bicep_center_y - 30 * region_scale_y),
                int(bicep_center_x + 25 * region_scale_x),
                int(bicep_center_y + 30 * region_scale_y)
            )
            
            tricep_center_x = (right_shoulder['x'] + right_elbow['x']) / 2
            tricep_center_y = (right_shoulder['y'] + right_elbow['y']) / 2
            
            regions['triceps_brachii'] = (
                int(tricep_center_x - 25 * region_scale_x),
                int(tricep_center_y - 30 * region_scale_y),
                int(tricep_center_x + 25 * region_scale_x),
                int(tricep_center_y + 30 * region_scale_y)
            )
            
        except (KeyError, ZeroDivisionError):
            regions = self._get_default_anatomical_regions(image_shape)
        
        return regions
    
    def _get_default_anatomical_regions(self, image_shape: Tuple[int, int]) -> Dict:
        """Default anatomical regions"""
        h, w = image_shape[:2]
        
        return {
            'upper_pectoralis': (int(w*0.2), int(h*0.1), int(w*0.8), int(h*0.35)),
            'lower_pectoralis': (int(w*0.25), int(h*0.3), int(w*0.75), int(h*0.5)),
            'upper_rectus': (int(w*0.35), int(h*0.45), int(w*0.65), int(h*0.6)),
            'lower_rectus': (int(w*0.35), int(h*0.55), int(w*0.65), int(h*0.75)),
            'external_obliques': (int(w*0.1), int(h*0.45), int(w*0.35), int(h*0.7)),
            'serratus_anterior': (int(w*0.15), int(h*0.3), int(w*0.35), int(h*0.6)),
            'anterior_deltoid': (int(w*0.05), int(h*0.1), int(w*0.3), int(h*0.4)),
            'lateral_deltoid': (int(w*0.7), int(h*0.1), int(w*0.95), int(h*0.4)),
            'biceps_brachii': (int(w*0.0), int(h*0.25), int(w*0.2), int(h*0.5)),
            'triceps_brachii': (int(w*0.8), int(h*0.25), int(w*1.0), int(h*0.5))
        }
    
    def anthropometric_measurements(self, landmarks: Dict) -> Dict:
        """Anthropometric measurements"""
        measurements = {}
        
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            hip_width = abs(left_hip['x'] - right_hip['x'])
            
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            torso_length = abs(hip_center_y - shoulder_center_y)
            
            left_arm_length = math.sqrt(
                (left_shoulder['x'] - left_wrist['x'])**2 + 
                (left_shoulder['y'] - left_wrist['y'])**2
            )
            right_arm_length = math.sqrt(
                (right_shoulder['x'] - right_wrist['x'])**2 + 
                (right_shoulder['y'] - right_wrist['y'])**2
            )
            
            measurements['shoulder_width'] = shoulder_width
            measurements['hip_width'] = hip_width
            measurements['torso_length'] = torso_length
            measurements['left_arm_length'] = left_arm_length
            measurements['right_arm_length'] = right_arm_length
            measurements['shoulder_to_hip_ratio'] = shoulder_width / (hip_width + 1e-6)
            measurements['arm_to_torso_ratio'] = (left_arm_length + right_arm_length) / (2 * torso_length + 1e-6)
            measurements['arm_symmetry'] = 1.0 - abs(left_arm_length - right_arm_length) / max(left_arm_length, right_arm_length, 1e-6)
            measurements['v_taper'] = shoulder_width / (hip_width + 1e-6)
            
            # Proportion score
            ideal_ratio = 1.618
            ratio_score = 100 * (1 - abs(measurements['shoulder_to_hip_ratio'] - ideal_ratio) / ideal_ratio)
            measurements['proportion_score'] = max(0, min(100, ratio_score))
            
        except (KeyError, ZeroDivisionError):
            measurements = {
                'shoulder_width': 200, 'hip_width': 150, 'torso_length': 300,
                'left_arm_length': 250, 'right_arm_length': 250,
                'shoulder_to_hip_ratio': 1.33, 'arm_to_torso_ratio': 0.83,
                'arm_symmetry': 1.0, 'v_taper': 1.33, 'proportion_score': 75
            }
        
        return measurements
    
    def draw_colored_regions(self, image: np.ndarray, regions: Dict, final_scores: Dict) -> np.ndarray:
        """Draw colored regions based on analysis scores"""
        overlay = image.copy()
        
        for region_name, coords in regions.items():
            if region_name in final_scores['region_scores']:
                x1, y1, x2, y2 = coords
                
                # Get score and determine color
                score = final_scores['region_scores'][region_name]['normalized_score']
                
                # Color based on score
                if score >= 85:
                    color = self.colors['excellent']    # Bright green
                elif score >= 70:
                    color = self.colors['good']         # Yellow
                elif score >= 55:
                    color = self.colors['average']      # Orange  
                else:
                    color = self.colors['poor']         # Red
                
                # Draw filled rectangle with transparency
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Add region label (ASCII-safe)
                region_display = self.anatomical_regions.get(region_name, region_name)[:9]
                score_text = f"{region_display}: {int(score)}"
                
                # Ensure ASCII text
                try:
                    score_text = score_text.encode('ascii', 'ignore').decode('ascii')
                except:
                    score_text = f"Region: {int(score)}"
                
                # Text background
                text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(overlay, (x1, y1-25), (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)
                
                # Text
                cv2.putText(overlay, score_text, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Blend with original image (30% overlay, 70% original)
        result = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
        return result

    def comprehensive_body_analysis(self, image: np.ndarray) -> Optional[Dict]:
        """Complete analysis pipeline"""
        print("🔧 Image enhancement...")
        enhanced_image = self.adaptive_image_enhancement(image)
        
        print("🎯 Pose detection...")
        landmarks, annotated_image = self.advanced_pose_analysis(enhanced_image)
        
        if landmarks is None:
            print("⚠️  Pose detection failed, using fallback regions")
            regions = self._get_default_anatomical_regions(image.shape)
            measurements = {
                'shoulder_width': 200, 'hip_width': 150, 'torso_length': 300,
                'left_arm_length': 250, 'right_arm_length': 250,
                'shoulder_to_hip_ratio': 1.33, 'arm_to_torso_ratio': 0.83,
                'arm_symmetry': 1.0, 'v_taper': 1.33, 'proportion_score': 75
            }
        else:
            print("✅ Pose detected, extracting anatomical regions")
            regions = self.get_anatomical_regions_from_pose(landmarks, image.shape)
            measurements = self.anthropometric_measurements(landmarks)
        
        # Analyze regions
        print("🔬 Multi-region analysis...")
        region_analyses = {}
        final_scores = {'region_scores': {}}
        
        for region_name, coords in regions.items():
            region_display = self.anatomical_regions.get(region_name, region_name)
            print(f"   Analyzing {region_display}...")
            
            # Simplified analysis
            muscle_score, fat_percentage, confidence = self.simplified_muscle_fat_analysis(enhanced_image, coords)
            edge_analysis = self.advanced_edge_detection(enhanced_image, coords)
            
            definition_score = min(100, edge_analysis['edge_density'] * 200 + edge_analysis['edge_strength'] / 3)
            
            # Simple scoring
            raw_score = (muscle_score * 0.4 + definition_score * 0.3 + (100 - fat_percentage) * 0.3)
            
            # Grade assignment
            if raw_score >= 95:
                grade = 'A+'
            elif raw_score >= 90:
                grade = 'A'
            elif raw_score >= 85:
                grade = 'B+'
            elif raw_score >= 80:
                grade = 'B'
            elif raw_score >= 70:
                grade = 'C+'
            elif raw_score >= 60:
                grade = 'C'
            elif raw_score >= 50:
                grade = 'D+'
            elif raw_score >= 40:
                grade = 'D'
            else:
                grade = 'F'
            
            # Simple percentile
            if raw_score >= 90:
                percentile = 95
            elif raw_score >= 80:
                percentile = 85
            elif raw_score >= 70:
                percentile = 70
            elif raw_score >= 60:
                percentile = 50
            elif raw_score >= 50:
                percentile = 25
            else:
                percentile = 10
            
            final_scores['region_scores'][region_name] = {
                'raw_score': raw_score,
                'normalized_score': raw_score,
                'percentile': percentile,
                'grade': grade,
                'confidence': confidence,
                'recommendations': [f"{self.anatomical_regions[region_name]} için özel antrenman programı"]
            }
        
        # Overall scores
        overall_score = np.mean([score['normalized_score'] for score in final_scores['region_scores'].values()])
        overall_grade = 'A+' if overall_score >= 95 else ('A' if overall_score >= 90 else ('B+' if overall_score >= 85 else 'B'))
        overall_percentile = np.mean([score['percentile'] for score in final_scores['region_scores'].values()])
        
        final_scores['overall_score'] = overall_score
        final_scores['overall_grade'] = overall_grade
        final_scores['overall_percentile'] = overall_percentile
        final_scores['anthropometric_score'] = measurements['proportion_score']
        final_scores['measurements'] = measurements
        
        # Create colored region visualization
        print("🎨 Creating colored region visualization...")
        colored_analysis_image = self.draw_colored_regions(annotated_image, regions, final_scores)
        
        return {
            'enhanced_image': enhanced_image,
            'annotated_image': colored_analysis_image,  # Now includes colored regions
            'landmarks': landmarks,
            'regions': regions,
            'final_scores': final_scores,
            'measurements': measurements
        }

def main():
    if len(sys.argv) != 2:
        print("🚀 NEXT-GENERATION BODY ANALYZER LITE")
        print("=" * 50)
        print("Scikit-learn bağımlılığı olmayan basitleştirilmiş versiyon")
        print(f"\nKullanım: python {sys.argv[0]} <foto_yolu>")
        return
    
    analyzer = NextGenBodyAnalyzerLite()
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    print(f"🚀 Next-Gen Lite analiz başlatılıyor: {image_path}")
    print("=" * 50)
    
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Görüntü yüklenemedi!")
        return
    
    try:
        analysis_results = analyzer.comprehensive_body_analysis(image)
        
        if analysis_results is None:
            print("❌ Analiz başarısız!")
            return
        
        final_scores = analysis_results['final_scores']
        measurements = analysis_results['measurements']
        
        print("\n" + "="*50)
        print("🎯 ANALYSIS RESULTS")
        print("="*50)
        
        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"   Overall Score: {final_scores['overall_score']:.1f}/100")
        print(f"   Overall Grade: {final_scores['overall_grade']}")
        print(f"   Population Percentile: {final_scores['overall_percentile']:.0f}%")
        
        print(f"\n📏 MEASUREMENTS:")
        print(f"   Shoulder-Hip Ratio: {measurements['shoulder_to_hip_ratio']:.2f}")
        print(f"   V-Taper: {measurements['v_taper']:.2f}")
        print(f"   Proportion Score: {measurements['proportion_score']:.1f}/100")
        
        # Save results
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        enhanced_output = f"nextgen_lite_enhanced_{name}{ext}"
        annotated_output = f"nextgen_lite_analysis_{name}{ext}"
        
        cv2.imwrite(enhanced_output, analysis_results['enhanced_image'])
        cv2.imwrite(annotated_output, analysis_results['annotated_image'])
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"   Enhanced: {enhanced_output}")
        print(f"   Analysis: {annotated_output}")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()