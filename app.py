"""AI-көмекші / ИИ-помощник для людей с ограниченными возможностями.
Запуск: streamlit run app.py
"""
import asyncio
import tempfile
from collections import Counter

import streamlit as st

T = {
    "kk": {
        "title": "AI-көмекші",
        "lang": "Тіл",
        "font": "Қаріп өлшемі",
        "contrast": "Жоғары контраст",
        "tabs": ["🎤 Сөйлеу → мәтін", "🔊 Мәтін → дыбыс", "🖼️ Сурет сипаттамасы"],
        "rec": "Микрофонға сөйлеңіз",
        "upload_audio": "немесе аудио файл жүктеңіз",
        "recognize": "Мәтінге айналдыру",
        "result": "Нәтиже",
        "listening": "Танылуда...",
        "text_in": "Оқылатын мәтінді енгізіңіз",
        "speak": "Дыбыстап оқу",
        "voice": "Дауыс",
        "upload_img": "Сурет жүктеңіз",
        "describe": "Суретті сипаттау",
        "loading": "Модель жұмыс істеуде...",
        "desc": "Сипаттама",
        "objects": "Табылған нысандар",
        "read_aloud": "Дыбыстап оқу",
        "no_input": "Алдымен деректерді енгізіңіз.",
        "err": "Қате",
        "scene": "Суретте",
    },
    "ru": {
        "title": "ИИ-помощник",
        "lang": "Язык",
        "font": "Размер шрифта",
        "contrast": "Высокая контрастность",
        "tabs": ["🎤 Речь → текст", "🔊 Текст → речь", "🖼️ Описание изображения"],
        "rec": "Говорите в микрофон",
        "upload_audio": "или загрузите аудиофайл",
        "recognize": "Преобразовать в текст",
        "result": "Результат",
        "listening": "Распознаю...",
        "text_in": "Введите текст для озвучивания",
        "speak": "Озвучить",
        "voice": "Голос",
        "upload_img": "Загрузите изображение",
        "describe": "Описать изображение",
        "loading": "Модель работает...",
        "desc": "Описание",
        "objects": "Найденные объекты",
        "read_aloud": "Озвучить",
        "no_input": "Сначала введите данные.",
        "err": "Ошибка",
        "scene": "На изображении",
    },
}

VOICES = {
    "kk": {"Айгүл (ә)": "kk-KZ-AigulNeural", "Дәулет (е)": "kk-KZ-DauletNeural"},
    "ru": {"Светлана (ж)": "ru-RU-SvetlanaNeural", "Дмитрий (м)": "ru-RU-DmitryNeural"},
}

st.set_page_config(page_title="AI Accessibility Assistant", page_icon="♿", layout="centered")

# ---------- Sidebar: accessibility settings ----------
lang = st.sidebar.radio("Тіл / Язык", ["kk", "ru"], format_func=lambda x: "Қазақша" if x == "kk" else "Русский")
t = T[lang]
font_px = st.sidebar.slider(t["font"], 16, 32, 22)
contrast = st.sidebar.toggle(t["contrast"], value=False)

bg, fg, accent = ("#000000", "#FFFF00", "#00FFFF") if contrast else ("#FFFFFF", "#111111", "#0B5FFF")
st.markdown(
    f"""
    <style>
    .stApp {{ background:{bg}; color:{fg}; }}
    html, body, p, label, span, div, textarea, input {{ font-size:{font_px}px !important; color:{fg}; }}
    h1 {{ font-size:{font_px + 14}px !important; color:{fg}; }}
    .stButton > button {{
        font-size:{font_px}px; padding:0.8em 1.4em; width:100%;
        background:{accent}; color:{'#000' if contrast else '#fff'};
        border:3px solid {fg}; border-radius:12px; font-weight:700;
    }}
    .stButton > button:focus {{ outline:4px solid orange; }}
    textarea {{ background:{bg} !important; color:{fg} !important; border:2px solid {fg} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♿ " + t["title"])


# ---------- Cached models ----------
@st.cache_resource(show_spinner=False)
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("small", compute_type="int8")


@st.cache_resource(show_spinner=False)
def get_captioner():
    from transformers import pipeline
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource(show_spinner=False)
def get_detector():
    from transformers import pipeline
    return pipeline("object-detection", model="facebook/detr-resnet-50")


# ---------- Helpers ----------
def transcribe(data: bytes, language: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    segments, _ = get_whisper().transcribe(path, language=language, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def tts(text: str, voice: str) -> bytes:
    import edge_tts

    async def run():
        out = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                out.extend(chunk["data"])
        return bytes(out)

    return asyncio.run(run())


def translate(text: str, target: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="en", target=target).translate(text)


def speak_widget(text: str, key: str):
    voices = VOICES[lang]
    name = st.selectbox(t["voice"], list(voices), key=f"v_{key}")
    if st.button("🔊 " + t["speak"], key=f"b_{key}"):
        if not text.strip():
            st.warning(t["no_input"])
        else:
            try:
                st.audio(tts(text, voices[name]), format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"{t['err']}: {e}")


tab_stt, tab_tts, tab_img = st.tabs(t["tabs"])

# ---------- 1. Speech -> text ----------
with tab_stt:
    audio = st.audio_input(t["rec"])
    uploaded = st.file_uploader(t["upload_audio"], type=["wav", "mp3", "m4a", "ogg"])
    src = audio or uploaded
    if st.button("📝 " + t["recognize"]):
        if src is None:
            st.warning(t["no_input"])
        else:
            try:
                with st.spinner(t["listening"]):
                    st.session_state["stt_text"] = transcribe(src.getvalue(), lang)
            except Exception as e:
                st.error(f"{t['err']}: {e}")
    if st.session_state.get("stt_text"):
        st.subheader(t["result"])
        st.text_area("", st.session_state["stt_text"], height=150, key="stt_out")
        speak_widget(st.session_state["stt_text"], "stt")

# ---------- 2. Text -> speech ----------
with tab_tts:
    txt = st.text_area(t["text_in"], height=200, key="tts_in")
    speak_widget(txt, "tts")

# ---------- 3. Image description ----------
with tab_img:
    img_file = st.file_uploader(t["upload_img"], type=["jpg", "jpeg", "png", "webp"])
    if img_file:
        from PIL import Image
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption=img_file.name, use_container_width=True)
        if st.button("👁️ " + t["describe"]):
            try:
                with st.spinner(t["loading"]):
                    caption = get_captioner()(image)[0]["generated_text"]
                    dets = [d["label"] for d in get_detector()(image) if d["score"] > 0.8]
                    counts = Counter(dets)
                    objs = ", ".join(f"{n} {lbl}" for lbl, n in counts.items()) or "-"
                    en = f"{caption}. Objects: {objs}."
                    full = translate(en, lang)
                st.session_state["img_text"] = full
            except Exception as e:
                st.error(f"{t['err']}: {e}")
    if st.session_state.get("img_text"):
        st.subheader(t["desc"])
        st.write(st.session_state["img_text"])
        speak_widget(st.session_state["img_text"], "img")
