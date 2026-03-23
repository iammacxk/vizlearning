import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from wordcloud import WordCloud
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
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

def is_valid_token(token):
    token = token.strip()
    if len(token) <= 1:
        return False
    if re.fullmatch(r'[\W_]+', token):
        return False
    if re.fullmatch(r'\d+', token):
        return False
    return True

def build_dynamic_stopwords(texts, base_stopwords, max_new_words=50):
    token_docs = []
    term_freq = Counter()
    doc_freq = defaultdict(int)

    for text in texts:
        tokens = [w for w in thai_tokenizer(str(text)) if is_valid_token(w)]
        token_docs.append(tokens)
        term_freq.update(tokens)
        for token in set(tokens):
            doc_freq[token] += 1

    if not token_docs:
        empty_df = pd.DataFrame(columns=['token', 'df_ratio', 'idf', 'chunk_ratio', 'tf', 'low_information_score', 'reason'])
        return set(base_stopwords), empty_df

    doc_count = len(token_docs)
    chunk_count = min(5, max(1, doc_count))
    chunk_size = max(1, int(np.ceil(doc_count / chunk_count)))
    chunk_freq = defaultdict(int)

    for i in range(chunk_count):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, doc_count)
        if start >= doc_count:
            continue
        chunk_tokens = set()
        for tokens in token_docs[start:end]:
            chunk_tokens.update(tokens)
        for token in chunk_tokens:
            chunk_freq[token] += 1

    frequent_tokens = {word for word, freq in doc_freq.items() if (freq / doc_count) >= 0.6}

    flattened = [" ".join(tokens) for tokens in token_docs if tokens]
    low_idf_tokens = set()
    idf_map = {}

    if flattened:
        vec = TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None)
        vec.fit_transform(flattened)
        feature_names = np.array(vec.get_feature_names_out())
        idf_scores = vec.idf_
        idf_map = {w: v for w, v in zip(feature_names, idf_scores)}
        idf_threshold = np.quantile(idf_scores, 0.2)
        low_idf_tokens = set(feature_names[idf_scores <= idf_threshold])

    protected_words = {
        'ดี', 'แย่', 'ยาก', 'ง่าย', 'ชอบ', 'ไม่ชอบ', 'สนุก', 'น่าเบื่อ', 'ล่ม', 'ช้า',
        'เข้าใจ', 'ไม่เข้าใจ', 'เครียด', 'ประทับใจ', 'พัฒนา', 'ปัญหา', 'คุณภาพ', 'เรียน',
        'อาจารย์', 'เนื้อหา', 'งาน', 'คะแนน', 'สอบ', 'หัวข้อ', 'ตัวอย่าง', 'บทเรียน'
    }

    candidate_tokens = set(frequent_tokens | low_idf_tokens)
    candidate_tokens = {w for w in candidate_tokens if is_valid_token(w) and w not in protected_words}

    diagnostics = []
    for token in candidate_tokens:
        df_ratio = doc_freq.get(token, 0) / doc_count
        idf_val = idf_map.get(token, 0.0)
        chunk_ratio = chunk_freq.get(token, 0) / chunk_count if chunk_count else 0
        tf_val = term_freq.get(token, 0)

        reason_flags = []
        if df_ratio >= 0.6:
            reason_flags.append('high_df')
        if token in low_idf_tokens:
            reason_flags.append('low_idf')
        if chunk_ratio >= 0.8:
            reason_flags.append('wide_coverage')

        low_information_score = (0.45 * df_ratio) + (0.35 * chunk_ratio) + (0.20 * (1.0 / max(idf_val, 1.0)))
        diagnostics.append({
            'token': token,
            'df_ratio': round(df_ratio, 3),
            'idf': round(float(idf_val), 3),
            'chunk_ratio': round(chunk_ratio, 3),
            'tf': int(tf_val),
            'low_information_score': round(float(low_information_score), 3),
            'reason': ','.join(reason_flags) if reason_flags else 'mixed'
        })

    diagnostics_df = pd.DataFrame(diagnostics)
    if diagnostics_df.empty:
        return set(base_stopwords), diagnostics_df

    diagnostics_df = diagnostics_df.sort_values(
        by=['low_information_score', 'tf', 'df_ratio'],
        ascending=[False, False, False]
    )

    learned_tokens = diagnostics_df.head(max_new_words)['token'].tolist()
    final_stopwords = set(base_stopwords) | set(learned_tokens)
    return final_stopwords, diagnostics_df

def tokenize_for_analysis(text, stopwords):
    tokens = thai_tokenizer(str(text))
    return [w for w in tokens if is_valid_token(w) and w not in stopwords]

def summarize_transcript_nlp(srt_df, stopwords, n_points=6):
    rows = srt_df[['Time', 'Text', 'Prediction_Class', 'Confidence']].copy()
    rows['Text'] = rows['Text'].astype(str).str.strip()
    rows = rows[rows['Text'] != '']
    if rows.empty:
        return {
            'summary_points': [],
            'key_terms': [],
            'actionable_insights': [],
            'summary_table': pd.DataFrame(columns=['Time', 'Text', 'Score'])
        }

    vec = TfidfVectorizer(
        tokenizer=thai_tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        stop_words=list(stopwords),
        min_df=1
    )
    matrix = vec.fit_transform(rows['Text'].tolist())
    if matrix.shape[0] == 1:
        only_text = rows.iloc[0]['Text']
        return {
            'summary_points': [f"[{rows.iloc[0]['Time']}] {only_text}"],
            'key_terms': [],
            'actionable_insights': [],
            'summary_table': pd.DataFrame([{
                'Time': rows.iloc[0]['Time'],
                'Text': only_text,
                'Score': 1.0
            }])
        }

    dense = matrix.toarray()
    centroid = dense.mean(axis=0, keepdims=True)
    relevance = cosine_similarity(dense, centroid).ravel()
    sim_matrix = cosine_similarity(dense)

    # MMR: เลือกประโยคสำคัญที่ยังไม่ซ้ำกันมาก เพื่อให้สรุปใช้งานได้จริง
    selected = []
    lambda_mmr = 0.72
    top_n = min(n_points, len(rows))
    candidate_idx = list(np.argsort(relevance)[::-1])

    while len(selected) < top_n and candidate_idx:
        if not selected:
            selected.append(candidate_idx.pop(0))
            continue

        best_idx = None
        best_score = -1e9
        for idx in candidate_idx:
            redundancy = max(sim_matrix[idx, s] for s in selected)
            mmr_score = (lambda_mmr * relevance[idx]) - ((1 - lambda_mmr) * redundancy)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        candidate_idx.remove(best_idx)

    selected = sorted(selected)
    selected_rows = rows.iloc[selected].copy()
    selected_rows['Score'] = [float(relevance[i]) for i in selected]

    summary_points = [f"[{r.Time}] {r.Text}" for r in selected_rows.itertuples()]

    feature_names = np.array(vec.get_feature_names_out())
    global_tfidf = dense.mean(axis=0)
    top_term_idx = np.argsort(global_tfidf)[::-1][:12]
    key_terms = [feature_names[i] for i in top_term_idx if is_valid_token(feature_names[i])]

    pos_rows = rows[rows['Prediction_Class'] == 2]
    neg_rows = rows[rows['Prediction_Class'] == 0]

    actionable_insights = []
    if not pos_rows.empty:
        best_pos = pos_rows.sort_values('Confidence', ascending=False).iloc[0]
        actionable_insights.append(f"ช่วงที่ให้ผลดี: [{best_pos['Time']}] {best_pos['Text']}")
    if not neg_rows.empty:
        best_neg = neg_rows.sort_values('Confidence', ascending=False).iloc[0]
        actionable_insights.append(f"ช่วงที่ควรปรับปรุง: [{best_neg['Time']}] {best_neg['Text']}")

    neg_terms = []
    if not neg_rows.empty:
        neg_tokens = []
        for text in neg_rows['Text']:
            neg_tokens.extend(tokenize_for_analysis(text, stopwords))
        neg_terms = [w for w, c in Counter(neg_tokens).most_common(5) if c >= 2]
        if neg_terms:
            actionable_insights.append(f"คำเสี่ยงที่พบซ้ำในบริบทลบ: {', '.join(neg_terms)}")

    return {
        'summary_points': summary_points,
        'key_terms': key_terms,
        'actionable_insights': actionable_insights,
        'summary_table': selected_rows[['Time', 'Text', 'Score']].reset_index(drop=True)
    }

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

# ----------------- Tab 1: Word Cloud -----------------
with tab1:
    st.header("ภาพรวมคำศัพท์ที่พูดถึงมากที่สุด (EDA)")
    if st.button("สร้าง Word Cloud"):
        with st.spinner("กำลังประมวลผลคำศัพท์..."):
            all_words = []
            base_stopwords = list(thai_stopwords()) + [' ', '  ', '\n', 'มาก', 'ดี', 'นี้', 'ก็', 'ๆ']
            stopwords, learned_sw_df = build_dynamic_stopwords(df['text'], base_stopwords)
            
            for text in df['text']:
                tokens = thai_tokenizer(text)
                words = [w for w in tokens if w not in stopwords and is_valid_token(w)]
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

                with st.expander("รายละเอียดคำที่ระบบเรียนรู้ว่าเป็น stop word (ชุดข้อมูลข้อเสนอแนะ)"):
                    if not learned_sw_df.empty:
                        st.dataframe(
                            learned_sw_df[['token', 'tf', 'df_ratio', 'chunk_ratio', 'idf', 'low_information_score', 'reason']].head(25),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("ยังไม่มีคำที่ระบบเรียนรู้เพิ่มเติม")
            except Exception as e:
                st.error(f"กรุณาตรวจสอบไฟล์ฟอนต์ภาษาไทย\nError: {e}")

# ----------------- Tab 2: LIME Visualization -----------------
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

# ----------------- Tab 3: SRT Analysis -----------------
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
                
                # ==========================================
                # ส่วนที่ 1: สรุปผลการวิเคราะห์ไฟล์บรรยายเชิงลึก
                # ==========================================
                st.divider()
                st.subheader("สรุปภาพรวมอารมณ์ของการบรรยาย (Detailed Summary)")
                
                total_lines = len(srt_df)
                pos_count = len(srt_df[srt_df['Prediction_Class'] == 2])
                neu_count = len(srt_df[srt_df['Prediction_Class'] == 1])
                neg_count = len(srt_df[srt_df['Prediction_Class'] == 0])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("จำนวนประโยคทั้งหมด", f"{total_lines} ประโยค")
                col2.metric("แง่บวก (Positive)", f"{(pos_count/total_lines)*100:.1f}%")
                col3.metric("เป็นกลาง (Neutral)", f"{(neu_count/total_lines)*100:.1f}%")
                col4.metric("แง่ลบ (Negative)", f"{(neg_count/total_lines)*100:.1f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**ประโยคแง่บวกที่ชัดเจนที่สุด 3 อันดับ:**")
                    top_pos = srt_df[srt_df['Prediction_Class'] == 2].sort_values(by='Confidence', ascending=False).head(3)
                    if not top_pos.empty:
                        st.dataframe(top_pos[['Time', 'Text']], hide_index=True, use_container_width=True)
                    else:
                        st.write("- ไม่พบประโยคแง่บวก -")
                        
                with colB:
                    st.markdown("**ประโยคแง่ลบที่ชัดเจนที่สุด 3 อันดับ:**")
                    top_neg = srt_df[srt_df['Prediction_Class'] == 0].sort_values(by='Confidence', ascending=False).head(3)
                    if not top_neg.empty:
                        st.dataframe(top_neg[['Time', 'Text']], hide_index=True, use_container_width=True)
                    else:
                        st.write("- ไม่พบประโยคแง่ลบ -")

                # ==========================================
                # ส่วนที่ 2: สรุปเนื้อหาด้วย NLP ภายในระบบ
                # ==========================================
                st.divider()
                st.subheader("สรุปใจความสำคัญจากวิดีโอด้วย NLP")
                st.markdown("ระบบสกัดประโยคสำคัญอัตโนมัติจากไฟล์ SRT ด้วย TF-IDF โดยไม่ต้องใช้ API ภายนอก")
                
                full_transcript = " ".join(srt_df['Text'].tolist())
                base_srt_stopwords = list(thai_stopwords()) + [
                    ' ', '  ', '\n', 'มาก', 'ดี', 'นี้', 'ก็', 'ๆ', 'ว่า', 'แล้ว', 'ครับ', 'ค่ะ',
                    'ให้', 'ได้', 'ที่', 'ไป', 'มา'
                ]
                srt_stopwords, srt_learned_sw_df = build_dynamic_stopwords(srt_df['Text'], base_srt_stopwords)

                if len(full_transcript) > 100:
                    with st.spinner("NLP กำลังสรุปประเด็นสำคัญจากไฟล์..."):
                        summary_result = summarize_transcript_nlp(srt_df, srt_stopwords, n_points=6)

                    st.success("สรุปผลอัตโนมัติสำเร็จ (ละเอียดเชิงใช้งาน)")
                    st.markdown("**ประเด็นสรุปหลัก (ลดความซ้ำและครอบคลุมหลายช่วงเวลา):**")
                    for idx, point in enumerate(summary_result['summary_points'], start=1):
                        st.write(f"{idx}. {point}")

                    st.markdown("**คำสำคัญที่เป็นแกนของเนื้อหา:**")
                    if summary_result['key_terms']:
                        st.write(", ".join(summary_result['key_terms']))
                    else:
                        st.write("ยังไม่พบคำสำคัญที่ชัดเจน")

                    st.markdown("**ข้อสรุปเชิงปฏิบัติ (Actionable):**")
                    if summary_result['actionable_insights']:
                        for insight in summary_result['actionable_insights']:
                            st.write(f"- {insight}")
                    else:
                        st.write("ยังไม่สามารถสกัดข้อสรุปเชิงปฏิบัติได้จากข้อมูลปัจจุบัน")

                    with st.expander("ตารางคะแนนประโยคที่ถูกเลือกเข้าสรุป"):
                        st.dataframe(summary_result['summary_table'], use_container_width=True, hide_index=True)

                    with st.expander("รายละเอียดคำที่ระบบเรียนรู้ว่าเป็น stop word (จากไฟล์ SRT นี้)"):
                        if not srt_learned_sw_df.empty:
                            st.dataframe(
                                srt_learned_sw_df[['token', 'tf', 'df_ratio', 'chunk_ratio', 'idf', 'low_information_score', 'reason']].head(35),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.write("ยังไม่มีคำที่ระบบเรียนรู้เพิ่มเติม")
                else:
                    st.write("เนื้อหาในวิดีโอสั้นเกินไป ระบบจึงข้ามการสรุปผล")

                # ==========================================
                # ส่วนที่ 3: Word Cloud จากไฟล์ SRT
                # ==========================================
                st.divider()
                st.subheader("ภาพรวมคำศัพท์จากบทบรรยาย (Transcript Word Cloud)")
                
                all_srt_words = []
                
                for text in srt_df['Text']:
                    tokens = thai_tokenizer(text)
                    words = [w for w in tokens if w not in srt_stopwords and is_valid_token(w)]
                    all_srt_words.extend(words)
                
                srt_word_freq = Counter(all_srt_words)
                
                if srt_word_freq:
                    try:
                        srt_wordcloud = WordCloud(
                            font_path=font_path, width=800, height=400, 
                            background_color='white', colormap='plasma' 
                        ).generate_from_frequencies(srt_word_freq)
                        
                        fig3, ax3 = plt.subplots(figsize=(10, 5))
                        ax3.imshow(srt_wordcloud, interpolation='bilinear')
                        ax3.axis("off")
                        st.pyplot(fig3)
                    except Exception as e:
                        st.error(f"ไม่สามารถสร้าง Word Cloud ได้ กรุณาตรวจสอบไฟล์ฟอนต์ภาษาไทย\nError: {e}")
                else:
                    st.warning("ไม่พบคำศัพท์ที่สามารถนำมาสร้าง Word Cloud ได้")

                # ==========================================
                # ส่วนที่ 4: กราฟ Timeline
                # ==========================================
                st.divider()
                st.subheader("กราฟแนวโน้มอารมณ์ของการบรรยาย (Sentiment Trajectory)")
                try:
                    thai_font = fm.FontProperties(fname=font_path, size=12)
                    thai_font_title = fm.FontProperties(fname=font_path, size=16)
                except:
                    thai_font = fm.FontProperties()
                    thai_font_title = fm.FontProperties()
                
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(srt_df.index, srt_df['Sentiment_Score'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
                
                ax2.set_yticks([-1, 0, 1])
                ax2.set_yticklabels(['แง่ลบ', 'เป็นกลาง', 'แง่บวก'], fontproperties=thai_font)
                
                ax2.set_title("ความเปลี่ยนแปลงทางอารมณ์ตามลำดับข้อความ", fontproperties=thai_font_title)
                ax2.set_xlabel("ลำดับข้อความในวิดีโอ (จากต้นจนจบ)", fontproperties=thai_font)
                ax2.set_ylabel("ระดับอารมณ์ความรู้สึก", fontproperties=thai_font)
                
                ax2.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig2)
                
                st.subheader("รายละเอียดการวิเคราะห์รายบรรทัด")
                st.dataframe(srt_df[['Time', 'Text', 'Prediction_Class', 'Confidence']], use_container_width=True)
                
            else:
                st.warning("ไม่พบรูปแบบข้อความที่ถูกต้องในไฟล์ SRT กรุณาตรวจสอบไฟล์อีกครั้ง")
