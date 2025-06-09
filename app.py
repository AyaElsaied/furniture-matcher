import gradio as gr
from PIL import Image
import shutil
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from skimage.color import rgb2hsv

# تحميل النماذج
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

model_yolo = YOLO("yolov8n.pt")

# إعداد المجلدات
os.makedirs("uploads", exist_ok=True)
os.makedirs("product_images", exist_ok=True)
os.makedirs("crops", exist_ok=True)

# تحميل بيانات المنتجات
df = pd.read_csv("furniture_products.csv")

# تحميل الصور وحفظها
def download_image(row):
    try:
        from io import BytesIO
        import requests
        response = requests.get(row["Image URL"])
        img = Image.open(BytesIO(response.content)).convert("RGB")
        filename = f"{row['ID']}.jpg"
        img.save(f"product_images/{filename}")
        return filename
    except:
        return None

df["filename"] = df.apply(download_image, axis=1)
df = df[df["filename"].notna()]

# استخراج الميزات
def extract_clip_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs).squeeze().numpy()
    return features / np.linalg.norm(features)

def extract_full_color_histogram(image_path):
    img = Image.open(image_path).convert('RGB').resize((100,100))
    img_np = np.array(img)
    hsv = rgb2hsv(img_np)
    h_hist, _ = np.histogram(hsv[:,:,0], bins=30, range=(0,1), density=True)
    s_hist, _ = np.histogram(hsv[:,:,1], bins=30, range=(0,1), density=True)
    v_hist, _ = np.histogram(hsv[:,:,2], bins=30, range=(0,1), density=True)
    return np.concatenate([h_hist, s_hist, v_hist])

def color_similarity(hist1, hist2):
    return np.minimum(hist1, hist2).sum()

# إعداد الفئات
category_weights = {
    "sofa": (0.6, 0.4),
    "chairs": (0.7, 0.3),
    "rug": (0.4, 0.6),
    "table": (0.8, 0.2)
}
category_map = {
    "couch": "sofa",
    "sofa": "sofa",
    "table": "table",
    "rug": "rug",
    "chair": "chairs"
}

def map_category(yolo_cat):
    return category_map.get(yolo_cat.lower(), None)

# استخراج الميزات للمنتجات
df["feature"] = df["filename"].apply(lambda f: extract_clip_features(f"product_images/{f}"))
df["color_hist"] = df["filename"].apply(lambda f: extract_full_color_histogram(f"product_images/{f}"))

# الوظيفة الرئيسية
def match_furniture(image):
    image_path = f"uploads/uploaded.jpg"
    image.save(image_path)

    results = model_yolo(image_path)
    img = np.array(Image.open(image_path).convert("RGB"))

    output = ""

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = img[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
        crop_path = f"crops/object_{i}.jpg"
        Image.fromarray(crop).save(crop_path)

        cls_id = int(results[0].boxes.cls[i])
        category_name = results[0].names[cls_id]
        mapped_cat = map_category(category_name)

        if not mapped_cat:
            continue

        alpha, beta = category_weights.get(mapped_cat, (0.7, 0.3))
        crop_feat = extract_clip_features(crop_path)
        crop_color = extract_full_color_histogram(crop_path)

        products_in_cat = df[df['Subcategory'].str.lower() == mapped_cat].copy()
        similarities = []

        for _, row in products_in_cat.iterrows():
            shape_sim = cosine_similarity([crop_feat], [row["feature"]])[0][0]
            color_sim = color_similarity(crop_color, row["color_hist"])
            combined_sim = alpha * shape_sim + beta * color_sim
            similarities.append(combined_sim)

        products_in_cat["similarity"] = similarities
        best_matches = products_in_cat.sort_values(by="similarity", ascending=False).head(3)

        output += f"\n### 🪑 أفضل تطابقات لفئة `{mapped_cat}`:\n"
        for _, row in best_matches.iterrows():
            output += f"- **{row['Name']}** - 💰 {row['Sale Price']} - 📷 [رابط الصورة]({row['Image URL']})\n"

    return output if output else "❌ لم يتم العثور على تطابقات واضحة."

# إنشاء الواجهة
interface = gr.Interface(
    fn=match_furniture,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="مطابقة الأثاث باستخدام الذكاء الاصطناعي",
    description="ارفع صورة لغرفة أو قطعة أثاث، وسنقوم بمطابقة الشكل واللون واقتراح منتجات مشابهة."
)

if __name__ == "__main__":
    interface.launch()
