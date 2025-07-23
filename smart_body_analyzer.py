#!/usr/bin/env python3
import cv2
import sys
import os
from simple_body_analyzer import SimpleBodyAnalyzer
from torso_analyzer import TorsoAnalyzer

def detect_image_type(image_path):
    """Fotoğrafın tam vücut mu yoksa üst vücut mu olduğunu tespit et"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    aspect_ratio = h / w
    
    print(f"📐 En-boy oranı: {aspect_ratio:.2f}")
    
    # Basit kural: En-boy oranı 1.5'den büyükse muhtemelen tam vücut
    if aspect_ratio > 1.5:
        return "full_body"
    else:
        return "torso"

def main():
    if len(sys.argv) != 2:
        print("🤖 AKILLI VÜCUT ANALİZİ")
        print("=" * 40)
        print("Kullanım:")
        print(f"  python {sys.argv[0]} <foto_yolu>")
        print("\nÖrnekler:")
        print(f"  python {sys.argv[0]} tam_vucut.jpg")
        print(f"  python {sys.argv[0]} ust_vucut.png")
        print("\nSistem otomatik olarak fotoğraf türünü algılar:")
        print("  🏃 Tam vücut → Detaylı pose analizi")  
        print("  💪 Üst vücut → Torso kompozisyon analizi")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Dosya bulunamadı: {image_path}")
        return
    
    print(f"🔬 Akıllı analiz başlatılıyor: {image_path}")
    print("-" * 50)
    
    # Fotoğraf türünü tespit et
    image_type = detect_image_type(image_path)
    
    if image_type is None:
        print("❌ Fotoğraf yüklenemedi!")
        return
    
    if image_type == "full_body":
        print("🏃 TAM VÜCUT tespit edildi - Pose analizi kullanılıyor...")
        analyzer = SimpleBodyAnalyzer()
        result = analyzer.analyze_image(image_path)
        
        if result[0] is not None:
            image, analysis, measurements = result
            output_path = f"fullbody_analysis_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, image)
            
            print(f"✅ BAŞARILI! Sonuç: {output_path}")
            
            if analysis:
                total_score = sum(r.get('score', 0) for r in analysis.values()) / len(analysis)
                print(f"🎯 Genel Skor: {total_score:.0f}/100")
                
                print("🏆 En iyi bölgeler:")
                sorted_regions = sorted(analysis.items(), key=lambda x: x[1].get('score', 0), reverse=True)
                for i, (region, data) in enumerate(sorted_regions[:3]):
                    print(f"  {i+1}. {analyzer.body_regions[region]}: {data['score']}/100")
        else:
            print("❌ Tam vücut pose tespit edilemedi, üst vücut analizine geçiliyor...")
            image_type = "torso"
    
    if image_type == "torso":
        print("💪 ÜST VÜCUT tespit edildi - Torso analizi kullanılıyor...")
        analyzer = TorsoAnalyzer()
        result = analyzer.analyze_image(image_path)
        
        if result is not None:
            image, regions = result
            output_path = f"torso_analysis_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, image)
            
            print(f"✅ BAŞARILI! Sonuç: {output_path}")
            
            total_score = sum(r['score'] for r in regions.values()) / len(regions)
            print(f"🎯 Genel Skor: {total_score:.0f}/100")
            
            # En iyi ve en zayıf bölgeler
            best_region = max(regions.items(), key=lambda x: x[1]['score'])
            worst_region = min(regions.items(), key=lambda x: x[1]['score'])
            
            print(f"🏆 En güçlü: {analyzer.regions[best_region[0]]} ({best_region[1]['score']}/100)")
            print(f"⚠️  Gelişim alanı: {analyzer.regions[worst_region[0]]} ({worst_region[1]['score']}/100)")
            
            # Genel değerlendirme
            if total_score >= 85:
                print("🎉 Harika üst vücut kompozisyonu!")
            elif total_score >= 70:
                print("👍 İyi gelişim gösteriyor, devam edin!")
            else:
                print("💪 Odaklanılacak alanlar var, planlı antrenman önerilir!")

if __name__ == "__main__":
    main()