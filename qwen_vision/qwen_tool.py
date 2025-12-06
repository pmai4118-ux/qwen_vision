import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from serpapi import GoogleSearch
import torch
from PIL import Image
import os

# --- CẤU HÌNH ---
st.set_page_config(page_title="Super AI Search (Qwen2-VL)", layout="wide", page_icon="🧠")

# Chặn GPU để tránh lỗi driver cũ, ép chạy CPU Ryzen 9 (Siêu mạnh)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# API Key SerpApi (Điền key của bạn vào đây)
DEFAULT_API_KEY = "f11961b7564ddf9c87c91417ebf19c612dce633ee4bb06829de5c83214475cb0" 

# --- 1. LOAD MODEL QWEN (Chỉ chạy 1 lần) ---
@st.cache_resource
def load_qwen_model():
    print("⏳ Đang tải Qwen2-VL-2B (Khoảng 3-4GB)...")
    # Load model bản 2B (Nhẹ, chạy tốt trên CPU)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    print("✅ Đã tải xong Qwen!")
    return model, processor

try:
    model, processor = load_qwen_model()
except Exception as e:
    st.error(f"Lỗi tải Model: {e}")
    st.stop()

# --- 2. HÀM XỬ LÝ ẢNH BẰNG AI ---
def analyze_image_with_qwen(image):
    # Prompt: Ra lệnh cho AI đóng vai chuyên gia tìm kiếm
    prompt_text = "Hãy đọc nội dung chữ trong ảnh này, nếu không có chữ, hãy nhận diện vật thể chính hoặc mô tả nội dung ảnh. và rút ra 1 từ khóa quan trọng nhất để tìm kiếm trên Google Images. Chỉ trả về từ khóa, không giải thích gì thêm."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Chuẩn bị dữ liệu cho AI
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Chạy AI để lấy kết quả
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Lấy phần trả lời của AI (cắt bỏ phần prompt)
    # Thường output sẽ là "User: ... \nAssistant: Highlands Coffee"
    # Ta lấy phần cuối cùng
    result = output_text[0].split("assistant\n")[-1].strip()
    return result

# --- 3. TÌM GOOGLE ---
def search_google(query, api_key):
    if not api_key: return []
    try:
        params = {"engine": "google_images", "q": query, "api_key": api_key, "num": 4}
        search = GoogleSearch(params)
        return search.get_dict().get("images_results", [])
    except: return []

# --- GIAO DIỆN WEB ---
st.title("🧠 Siêu AI Tìm Kiếm (Powered by Qwen2-VL)")
st.caption("Sử dụng Vision-Language Model để 'hiểu' ảnh thay vì chỉ đọc chữ.")

with st.sidebar:
    st.header("Cấu hình")
    api_key_input = st.text_input("SerpApi Key:", value=DEFAULT_API_KEY, type="password")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Upload ảnh...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh Input", use_container_width=True)
        
        if st.button("🚀 PHÂN TÍCH & TÌM KIẾM", type="primary"):
            with col2:
                with st.status("Đang suy nghĩ...", expanded=True):
                    st.write("🧠 Qwen đang nhìn ảnh...")
                    # Gọi AI
                    keyword = analyze_image_with_qwen(image)
                    
                    st.success("Đã hiểu ảnh!")
                    st.metric(label="Từ khóa trích xuất", value=keyword)
                    
                    if api_key_input:
                        st.write(f"🌍 Đang tìm Google: {keyword}...")
                        results = search_google(keyword, api_key_input)
                    else:
                        results = None

                if results:
                    st.divider()
                    st.subheader("Kết quả tìm kiếm:")
                    cols = st.columns(2)
                    for i, img in enumerate(results):
                        with cols[i%2]:
                            try:
                                st.image(img['original'], use_container_width=True)
                                st.caption(img['title'])
                            except: pass
# neu duoc tai tu may khac thi chay lenh nay trong dev terminal
# pip install streamlit transformers pillow torch serpapi
# de chay duoc thi go trong terminal
# streamlit run qwen_vision/qwen_tool.py
