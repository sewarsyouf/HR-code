import os
import re
import shutil
import json
from collections import Counter
from datetime import datetime
from tqdm import tqdm

import cv2
import pytesseract
import pandas as pd

# -------------------------
# إعداد مسار tesseract إذا لزم
# (مثال لويندوز) عدّل المسار ليتوافق مع جهازك
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -------------------------

# ======= قائمة كلمات مفتاحية (قابلة للتعديل) ========
DEGREE_KEYWORDS = {
    "phd": ["phd", "ph.d", "doctor of philosophy", "دكتور"],
    "master": ["master", "msc", "m.sc", "ma", "m.a", "ماجستير", "ماجسـتير"],
    "bachelor": ["bachelor", "b.sc", "bsc", "b.a", "بكالوريوس"],
}

EXPERIENCE_WORDS = ["experience", "years", "years of experience", "خبرة", "سنة", "سنوات", "worked", "employment"]
SKILLS_LIST = [
    # ضع هنا المهارات المتوقعة لوظيفة معينة؛ يمكن استبدالها أو توسعتها
    "python", "java", "c++", "machine learning", "deep learning",
    "sql", "excel", "power bi", "tensorflow", "pytorch",
    "communication", "teamwork", "aws", "azure", "docker",
    "git", "linux", "html", "css", "javascript",
    "data analysis", "data science", "nlp", "computer vision",
    "project management", "leadership"
]
CERT_KEYWORDS = ["certificate", "certified", "certification", "دورة", "شهادة"]

CONTACT_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(\+?\d{7,15})"
}
LINK_PATTERNS = ["linkedin", "github", "portfolio", "behance"]

# ======= وظائف المساعدة ========
def extract_text_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return ""
    # تحويل إلى رمادي وتهيئة بسيطة لOCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # يمكن استخدام threshold / denoise إذا احتجت
    text = pytesseract.image_to_string(gray, lang='eng+ara')  # استخدم arabic إذا النص عربي
    return text

def count_years_experience(text):
    # محاولة استخلاص سنوات الخبرة باستخدام تعابير بسيطة
    # نبحث عن أرقام متبوعة بكلمة سنة/years
    matches = re.findall(r"(\d{1,2})\s*(?:years|year|ي?سنة|سنوات)", text, flags=re.IGNORECASE)
    nums = [int(m) for m in matches] if matches else []
    return max(nums) if nums else 0

def detect_highest_degree(text):
    t = text.lower()
    for deg, kws in DEGREE_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return deg
    return "none"

def count_skills(text, skills_list=SKILLS_LIST):
    t = text.lower()
    found = []
    for s in skills_list:
        if s.lower() in t:
            found.append(s)
    return found

def detect_contact(text):
    contacts = {}
    for k, pat in CONTACT_PATTERNS.items():
        m = re.search(pat, text)
        contacts[k] = m.group(0) if m else None
    # links
    tl = text.lower()
    contacts["has_link"] = any(l in tl for l in LINK_PATTERNS)
    return contacts

def count_certifications(text):
    t = text.lower()
    cnt = sum(1 for kw in CERT_KEYWORDS if kw in t)
    return cnt

# ======= دالة تقييم بسيطة (قابلة للتعديل) ========
def score_features(features, weights=None):
    """
    features: dict with keys:
      'years', 'degree' ('phd','master','bachelor','none'),
      'n_skills', 'has_contact', 'n_certs', 'text_len'
    weights: dict of weights
    """
    if weights is None:
        weights = {
            "years": 1.5,
            "degree_phd": 2.5,
            "degree_master": 1.5,
            "degree_bachelor": 1.0,
            "n_skills": 1.0,
            "contact": 1.0,
            "cert": 0.8,
            "length": 0.3
        }
    score = 0.0

    # سنوات الخبرة (نطاقية)
    score += min(features["years"], 20) * weights["years"] / 5.0  # نطبع تأثيرها

    # الدرجة العلمية
    deg = features["degree"]
    if deg == "phd":
        score += weights["degree_phd"]
    elif deg == "master":
        score += weights["degree_master"]
    elif deg == "bachelor":
        score += weights["degree_bachelor"]

    # عدد المهارات
    score += features["n_skills"] * weights["n_skills"]

    # وجود معلومات تواصل
    score += (1 if features["has_contact"] else 0) * weights["contact"]

    # شهادات
    score += min(features["n_certs"], 5) * weights["cert"]

    # طول النص (كمؤشر على تفصيل)
    score += min(features["text_len"] / 1000.0, 2.0) * weights["length"]

    return round(score, 3)

# ======= الدالة الرئيسية ========
def process_folder(folder_path, out_csv="cv_scores.csv", out_sorted_folder="sorted_cvs", use_openai=False):
    records = []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    os.makedirs(out_sorted_folder, exist_ok=True)

    for fname in tqdm(files, desc="Processing images"):
        path = os.path.join(folder_path, fname)
        text = extract_text_from_image(path)
        text_len = len(text)

        years = count_years_experience(text)
        degree = detect_highest_degree(text)
        skills = count_skills(text)
        contact = detect_contact(text)
        n_certs = count_certifications(text)

        features = {
            "file": fname,
            "years": years,
            "degree": degree,
            "n_skills": len(skills),
            "skills_list": skills,
            "has_contact": bool(contact.get("email") or contact.get("phone") or contact.get("has_link")),
            "contacts": contact,
            "n_certs": n_certs,
            "text_len": text_len,
            "raw_text_snippet": text[:1000].replace("\n"," "),
        }
        score = score_features(features)
        features["score"] = score
        records.append(features)

    # ترتيب النتائج تنازليًا (الأعلى أولًا)
    records_sorted = sorted(records, key=lambda x: x["score"], reverse=True)

    # نسخ الصور إلى مجلد مرتب مع بادئة ترتيب
    for idx, rec in enumerate(records_sorted, start=1):
        src = os.path.join(folder_path, rec["file"])
        ext = os.path.splitext(rec["file"])[1]
        dst_name = f"{idx:02d}_{rec['score']}_{rec['file']}"
        dst = os.path.join(out_sorted_folder, dst_name)
        shutil.copyfile(src, dst)

    # حفظ CSV
    df = pd.DataFrame(records_sorted)
    df.to_csv(out_csv, index=False)

    # حفظ JSON مختصر
    with open("cv_scores.json", "w", encoding="utf-8") as f:
        json.dump(records_sorted, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to {out_csv} and images copied to {out_sorted_folder}")
    return records_sorted

# ======= تنفيذ (تعديل المسار حسب حاجتك) ========
if __name__ == "__main__":
    folder = "cvs_images"   # ضع هنا مسار مجلد الصور
    results = process_folder(folder)
    # طباعة الخمسة الأوائل
    for r in results[:5]:
        print(r['file'], r['score'], r['years'], r['degree'], r['n_skills'])
