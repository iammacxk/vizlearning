import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import streamlit.components.v1 as components

# ==========================================
# 1. เตรียมข้อมูลพื้นฐานสำหรับฝึกสอนโมเดล
# ==========================================
base_samples = [
    ('ครูสอนดีมาก เข้าใจง่ายสุดๆ', 2),
    ('ระบบใช้งานยากมาก กระตุกบ่อยรำคาญ', 0),
    ('เนื้อหาน่าสนใจดี แต่การบ้านเยอะเกินไป', 1),
    ('แอปนี้ช่วยให้ติดตามงานได้ดีขึ้นเยอะ', 2),
    ('แย่มาก โหลดอะไรไม่ค่อยขึ้นเลย', 0),
    ('ชอบมากครับ ใช้งานสะดวกดี', 2),
    ('ก็งั้นๆ ไม่ได้มีอะไรพิเศษ', 1),
    ('ระบบดีนะ ดีออก... (ประชด)', 0),
    ('วิชานี้สนุกมาก อาจารย์ใจดี', 2),
    ('สอบยากเกินไป ทำไม่ทันเลย', 0),
    ('ระบบช้ามาก ส่งงานไม่ทันเพราะระบบล่ม', 0),
    ('เนื้อหาดี แต่ระบบควรปรับปรุง', 1),
]

additional_samples = [
    ('อาจารย์อธิบายละเอียดมาก เรียนแล้วเข้าใจทันที', 2),
    ('สไลด์อ่านยาก ตัวหนังสือเล็กเกินไป', 0),
    ('เนื้อหาโอเค แต่จังหวะสอนค่อนข้างเร็ว', 1),
    ('ระบบส่งงานล่มบ่อย ทำให้เครียดมาก', 0),
    ('กิจกรรมในคลาสสนุกและได้ลงมือทำจริง', 2),
    ('เฉยๆ ยังไม่ค่อยเห็นความต่างจากวิชาอื่น', 1),
    ('ชอบที่มีตัวอย่างจริง ทำให้เห็นภาพชัด', 2),
    ('คำอธิบายไม่ต่อเนื่อง ทำให้งงหลายช่วง', 0),
    ('ภาพรวมถือว่าใช้ได้ ไม่มีอะไรโดดเด่น', 1),
    ('ผู้สอนเป็นกันเองและตอบคำถามเร็วมาก', 2),
    ('ระบบแจ้งเตือนช้า พลาดกำหนดส่งหลายครั้ง', 0),
    ('เนื้อหาพอใช้ แต่แบบฝึกหัดค่อนข้างซ้ำ', 1),
    ('เรียนแล้วได้ทักษะเพิ่มขึ้นอย่างชัดเจน', 2),
    ('ตารางเรียนแน่นเกินไป เหนื่อยมาก', 0),
    ('เนื้อหากลางๆ ไม่ง่ายไม่ยากเกินไป', 1),
    ('แอปเสถียรขึ้นมาก ใช้งานลื่นดี', 2),
    ('โหลดไฟล์ช้ามาก เสียเวลาไปเยอะ', 0),
    ('บทเรียนพอเข้าใจ แต่ยังต้องทบทวนอีก', 1),
    ('อาจารย์ให้ฟีดแบ็กละเอียดและมีประโยชน์', 2),
    ('ระบบชอบเด้งออกตอนกำลังพิมพ์งาน', 0),
    ('โดยรวมกลางๆ ไม่มีปัญหาร้ายแรง', 1),
    ('ชอบรูปแบบการสอนที่มี workshop', 2),
    ('ตัวอย่างน้อยไป ทำให้ประยุกต์ยาก', 0),
    ('เนื้อหาพื้นฐานค่อนข้างโอเค', 1),
    ('เรียนสนุก ได้ทั้งความรู้และแรงบันดาลใจ', 2),
    ('งานกลุ่มเยอะเกินไป ประสานงานลำบาก', 0),
    ('คุณภาพเสียงในวิดีโอพอใช้ได้', 1),
    ('มีแบบฝึกหัดให้ลองทำทันที ดีมาก', 2),
    ('บางหัวข้ออธิบายสั้นเกินไปตามไม่ทัน', 0),
    ('วิชานี้ถือว่าโอเคตามมาตรฐาน', 1),
    ('เนื้อหาทันสมัยและนำไปใช้จริงได้', 2),
    ('ระบบค้นหาเอกสารใช้งานยากมาก', 0),
    ('โดยรวมไม่มีอะไรแย่ แต่ยังไม่ว้าว', 1),
    ('ชอบที่สรุปท้ายคาบ ทำให้จำได้ดี', 2),
    ('ส่งงานแล้วระบบไม่ขึ้นสถานะ ทำให้กังวล', 0),
    ('ระดับความยากปานกลาง กำลังดี', 1),
]

all_samples = base_samples + additional_samples
df = pd.DataFrame(all_samples, columns=['text', 'label'])

# ==========================================
# 2. ฟังก์ชัน NLP, AI และการจัดการไฟล์
# ==========================================
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm')

@st.cache_resource
def train_model():
    vec = TfidfVectorizer(tokenizer=thai_tokenizer, ngram_range=(1, 2))
    clf = LogisticRegression(random_state=42, max_iter=1000)
    pipeline = make_pipeline(vec, clf)
    pipeline.fit(df['text'], df['label'])
    return pipeline

pipeline = train_model()
explainer = LimeTextExplainer(class_names=['Negative', 'Neutral', 'Positive'], split_expression=thai_tokenizer)

def parse_srt(file_content):
    blocks = file_content.strip().split('\n\n')
    parsed_data = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            time_match = re.search(r'(\d{2}:\d{2}:\d{2})', lines[1])
            start_time = time_match.group(1) if time_match else "00:00:00"
            text = " ".join(lines[2:])
            text = re.sub(r'<[^>]+>', '', text)
            if text.strip():
                parsed_data.append({"Time": start_time, "Text": text.strip()})
    return pd.DataFrame(parsed_data)

# กำหนดเส้นทางไฟล์ฟอนต์ภาษาไทย (ใช้ร่วมกันทั้ง Word Cloud และ Matplotlib)
font_path = 'THSarabunNew.ttf'

# ==========================================
# 3. ส่วนแสดงผล Web App (UI)
# ==========================================
st.set_page_config(page_title="Student Feedback & Transcript Analyzer", layout="wide")
st.title("ระบบวิเคราะห์ข้อเสนอแนะและบทบรรยายการสอน")

tab1, tab2, tab3 = st.tabs([
    "ภาพรวมข้อเสนอแนะ (Word Cloud)", 
    "วิเคราะห์ข้อความรายบุคคล (LIME)", 
    "วิเคราะห์ไฟล์บรรยาย SRT (Timeline)"
])

with tab1:
    st.header("ภาพรวมคำศัพท์ที่พูดถึงมากที่สุด (EDA)")
    if st.button("สร้าง Word Cloud"):
        with st.spinner("กำลังประมวลผลคำศัพท์..."):
            all_words = []
            stopwords = list(thai_stopwords()) + [' ', '  ', '\n', 'มาก', 'ดี', 'นี้', 'ก็', 'ๆ'] 
            
            for text in df['text']:
                tokens = thai_tokenizer(text)
                words = [w for w in tokens if w not in stopwords and len(w) > 1]
                all_words.extend(words)
            
            word_freq = Counter(all_words)
            try:
                wordcloud = WordCloud(
                    font_path=font_path, width=800, height=400, 
                    background_color='white', colormap='viridis'
                ).generate_from_frequencies(word_freq)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"กรุณาตรวจสอบไฟล์ฟอนต์ภาษาไทย\nError: {e}")

with tab2:
    st.header("วิเคราะห์อารมณ์และอรรถาธิบายการทำงานของแบบจำลอง (LIME)")
    user_input = st.text_area("กรอกข้อความที่ต้องการวิเคราะห์:", "วิชานี้เนื้อหาดีมาก แต่ระบบส่งงานใช้งานยากไปหน่อย")

    if st.button("วิเคราะห์และแสดงผล (Analyze & Visualize)"):
        with st.spinner('กำลังวิเคราะห์ข้อมูล...'):
            prediction = pipeline.predict([user_input])[0]
            prob = pipeline.predict_proba([user_input])[0] 
            
            result_map = {0: ("แง่ลบ (Negative)", "red"), 1: ("เป็นกลาง (Neutral)", "orange"), 2: ("แง่บวก (Positive)", "green")}
            result_text, result_color = result_map[prediction]
            
            st.markdown(f"### ผลการทำนายหลัก: <span style='color:{result_color}'>{result_text}</span>", unsafe_allow_html=True)
            st.subheader("LIME Visualization (การวิเคราะห์น้ำหนักคำศัพท์)")
            exp = explainer.explain_instance(user_input, pipeline.predict_proba, num_features=6, labels=[prediction])
            components.html(exp.as_html(), height=400, scrolling=True)

with tab3:
    st.header("วิเคราะห์แนวโน้มอารมณ์จากไฟล์คำบรรยาย (.srt)")
    st.markdown("ระบบจะสกัดข้อความตามช่วงเวลาและวิเคราะห์การเปลี่ยนแปลงของบริบทแบบ Time-Series")
    
    uploaded_file = st.file_uploader("อัปโหลดไฟล์คำบรรยาย (.srt)", type=['srt'])
    
    if uploaded_file is not None:
        with st.spinner('กำลังอ่านและประมวลผลไฟล์...'):
            content = uploaded_file.read().decode("utf-8")
            srt_df = parse_srt(content)
            
            if not srt_df.empty:
                predictions = pipeline.predict(srt_df['Text'])
                probabilities = pipeline.predict_proba(srt_df['Text'])
                
                srt_df['Prediction_Class'] = predictions
                srt_df['Sentiment_Score'] = srt_df['Prediction_Class'].map({0: -1, 1: 0, 2: 1})
                srt_df['Confidence'] = np.max(probabilities, axis=1)
                
                st.success(f"ประมวลผลสำเร็จ จำนวนข้อความทั้งหมด: {len(srt_df)} บรรทัด")
                
                # ตั้งค่าคุณสมบัติฟอนต์สำหรับ Matplotlib
                try:
                    thai_font = fm.FontProperties(fname=font_path, size=12)
                    thai_font_title = fm.FontProperties(fname=font_path, size=16)
                except:
                    # กรณีหาไฟล์ฟอนต์ไม่พบ จะใช้ค่าเริ่มต้นเพื่อไม่ให้ระบบพัง
                    thai_font = fm.FontProperties()
                    thai_font_title = fm.FontProperties()
                
                # แสดงกราฟ Timeline พร้อมฟอนต์ภาษาไทย
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(srt_df.index, srt_df['Sentiment_Score'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
                
                ax2.set_yticks([-1, 0, 1])
                # เปลี่ยนแกน Y เป็นภาษาไทย
                ax2.set_yticklabels(['แง่ลบ', 'เป็นกลาง', 'แง่บวก'], fontproperties=thai_font)
                
                ax2.set_title("แนวโน้มอารมณ์ของการบรรยาย (Sentiment Trajectory)", fontproperties=thai_font_title)
                ax2.set_xlabel("ลำดับข้อความในวิดีโอ (ตามเวลา)", fontproperties=thai_font)
                ax2.set_ylabel("ระดับอารมณ์ความรู้สึก", fontproperties=thai_font)
                
                ax2.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig2)
                
                st.subheader("รายละเอียดการวิเคราะห์รายบรรทัด")
                st.dataframe(srt_df[['Time', 'Text', 'Prediction_Class', 'Confidence']], use_container_width=True)
                
            else:
                st.warning("ไม่พบรูปแบบข้อความที่ถูกต้องในไฟล์ SRT กรุณาตรวจสอบไฟล์อีกครั้ง")
