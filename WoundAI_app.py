import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# --- การตั้งค่าหน้าเว็บและการแสดงผลแบบเต็มหน้าจอ (Wide Layout) ---
st.set_page_config(page_title="WoundAi - Analysis", layout="wide")

# สไตล์ CSS หลัก สำหรับปรับแต่งสตรีมลิตให้สวยงามเต็มหน้าจออย่างแท้จริง
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
    <style>
        body, .main, .block-container {
            background-color: #f6faff !important;
            font-family: 'Inter', sans-serif;
            padding-top: 10px !important;
            max-width: 100% !important; 
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .center-text {
            text-align: center;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #e6eff8;
            border-radius: 8px 8px 0px 0px;
            padding: 8px 16px;
            color: #424752;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00478d !important;
            color: white !important;
        }
        .guideline-box {
            background-color: #ecf5fe;
            border: 1px solid #c2c6d4;
            border-radius: 12px;
            padding: 16px;
            margin-top: 25px;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- คลังข้อมูลวิธีปฐมพยาบาล (ลบคีย์สะกดผิดและคีย์ซ้ำซ้อนออกแล้ว) ---
FIRST_AID_GUIDE = {
    "abrasion_wound": {
        "title": "Abrasion Wound", "th_title": "แผลถลอก", "is_normal": False, "badge": "Surface Injury",
        "findings": "พบรอยโรคที่มีลักษณะเป็นแผลตื้นบริเวณผิวหนังกำพร้า มีการถลอกจากการครูดหรือเสียดสี",
        "morphology": "มีเลือดออกซึมเล็กน้อย ขอบแผลไม่เรียบ พื้นแผลมีสีแดงเรื่อ อาจมีเศษดินหรือสิ่งสกปรกปนเปื้อนเล็กน้อย",
        "steps": [
            "ล้างแผลให้สะอาดด้วยน้ำเกลือปราศจากเชื้อ (Saline) หรือเปิดน้ำสะอาดไหลผ่านเบาๆ เพื่อล้างเศษดินและสิ่งสกปรกออก",
            "ทายาฆ่าเชื้อหรือยาปฏิชีวนะบางๆ บนผิวแผล เพื่อป้องกันการติดเชื้อและรักษาความชุ่มชื้นของเนื้อเยื่อ",
            "ปิดแผลด้วยผ้าก๊อซหรือพลาสเตอร์ชนิดไม่ติดแผล (Non-stick pad) เพื่อป้องกันสิ่งสกปรกและการเสียดสีภายนอก"
        ]
    },
    "bruises_wound": {
        "title": "Bruises Wound", "th_title": "แผลฟกช้ำ", "is_normal": False, "badge": "Hematoma",
        "findings": "พบการสะสมของเลือดใต้ชั้นผิวหนังจากการกระแทก โดยไม่มีรอยฉีกขาดของผิวหนังภายนอก",
        "morphology": "ผิวหนังมีสีม่วง คล้ำ เขียว หรือเหลืองตามอายุของรอยช้ำ มีอาการบวมและเจ็บเสียวเมื่อกด",
        "steps": [
            "ประคบเย็นบริเวณที่บวมช้ำด้วยเจลเย็นหรือถุงน้ำแข็งห่อผ้า นาน 15-20 นาที ทุกๆ 2-3 ชั่วโมง ในช่วง 48 ชั่วโมงแรกเพื่อลดบวม",
            "ยกส่วนที่บาดเจ็บหรืออวัยวะนั้นๆ ให้สูงกว่าระดับหัวใจเท่าที่ทำได้ และพักการใช้งานเพื่อลดการสะสมของเลือดใต้ผิวหนัง",
            "เมื่อพ้น 48 ชั่วโมงแรกไปแล้ว ให้เปลี่ยนเป็นประคบอุ่นเพื่อกระตุ้นการไหลเวียนเลือดและเร่งการดูดซึมรอยช้ำให้หายเร็วขึ้น"
        ]
    },
    "burn_wound": {  
        "title": "Burn Wound", "th_title": "แผลไฟไหม้ / น้ำร้อนลวก", "is_normal": False, "badge": "Thermal Injury",
        "findings": "พบการทำลายของเนื้อเยื่อผิวหนังจากความร้อน สารเคมี หรือกระแสไฟฟ้า",
        "morphology": "ผิวหนังแดงจัด บวม มีตุ่มน้ำพอง (Blisters) หรือผิวหนังลอกหลุด มีความรู้สึกปวดแสบปวดร้อนรุนแรง",
        "steps": [
            "ล้างแผลทันทีด้วยน้ำสะอาดอุณหภูมิห้องไหลผ่านเบาๆ นาน 10-20 นาที เพื่อระบายความร้อน ห้ามใช้น้ำแข็งหรือยาสีฟันเด็ดขาด",
            "ห้ามเจาะหรือแกะตุ่มน้ำพองที่เกิดขึ้นเอง เนื่องจากผิวหนังของตุ่มน้ำเป็นเกราะป้องกันเชื้อโรคตามธรรมชาติที่สำคัญที่สุด",
            "ทาเจลว่านหางจระเข้สำหรับแผลไหม้หรือยาทางแพทย์ จากนั้นปิดแผลหลวมๆ ด้วยผ้าก๊อซสะอาดเพื่อป้องกันสิ่งสกปรก"
        ]
    },
    "cut_wound": {
        "title": "Cut Wound", "th_title": "แผลถูกบาด / แผลของมีคม", "is_normal": False, "badge": "Incised Wound",
        "findings": "พบรอยแยกของผิวหนังเป็นทางยาวขอบเรียบ เกิดจากของมีคมบาด",
        "morphology": "ปากแผลเรียบชิดติดกันหรือแยกออกเล็กน้อย มีเลือดไหลค่อนข้างมาก ความลึกขึ้นอยู่กับแรงกด",
        "steps": [
            "ใช้ผ้าสะอาดหรือผ้าก๊อซกดลงบนบาดแผลโดยตรงอย่างต่อเนื่องเป็นเวลา 2-5 นาทีเพื่อห้ามเลือดให้สนิทก่อน",
            "ล้างแผลด้วยน้ำสะอาดและสบู่บริเวณรอบๆ แผล หลีกเลี่ยงการเทแอลกอฮอล์ใส่ในแผลโดยตรงเพราะจะทำให้เนื้อเยื่อแสบไหม้",
            "ทายาฆ่าเชื้อรักษาสมานแผล ดึงขอบแผลให้ชิดเข้าหากัน แล้วปิดด้วยพลาสเตอร์ยาหรือผ้าก๊อซให้แน่นกระชับ"
        ]
    },
    "diabetic_wound": {
        "title": "Diabetic Wound", "th_title": "แผลเบาหวาน", "is_normal": False, "badge": "Chronic Ulcer",
        "findings": "พบแผลเรื้อรังบริเวณส่วนปลาย (มักเป็นที่เท้า) ในผู้ป่วยที่มีประวัติโรคเบาหวาน",
        "morphology": "ขอบแผลหนาและแข็ง (Callus) พื้นแผลมักลึก การไหลเวียนเลือดส่วนปลายลดลง อาจมีความรู้สึกชา",
        "steps": [
            "ล้างแผลเบาๆ ด้วยน้ำเกลือทางการแพทย์เท่านั้น ห้ามใช้ยาฆ่าเชื้อที่รุนแรง เช่น เบตาดีนเข้มข้นหรือแอลกอฮอล์ในแผลเด็ดขาด",
            "ใช้แผ่นปิดแผลเฉพาะทางที่ช่วยรักษาความชุ่มชื้น (เช่น Hydrocolloid หรือ Foam) ตรวจสอบแผลและสัญญาณติดเชื้อทุกวัน",
            "ปฏิบัติตามหลัก Offloading อย่างเคร่งครัด (ห้ามเดินลงน้ำหนักบริเวณที่เป็นแผล) และควรรีบไปพบแพทย์เฉพาะทาง"
        ]
    },
    "laceration_wound": { 
        "title": "Laceration Wound", "th_title": "แผลฉีกขาดฉกรรจ์", "is_normal": False, "badge": "Trauma Injury",
        "findings": "พบแผลฉีกขาดขอบไม่เรียบขรุขระจากการถูกของแข็งกระแทกหรือฉีกกระชาก",
        "morphology": "เนื้อเยื่อมีการฉีกขาดรุ่งริ่ง แผลมักจะลึกและกว้าง มีเลือดออกปานกลางถึงมาก มักมีสิ่งปนเปื้อน",
        "steps": [
            "กดห้ามเลือดทันที โดยใช้ผ้าก๊อซหนาๆ หรือผ้าสะอาดกดลงบนแผลให้แน่นและนิ่งที่สุด หากเลือดซึมเพิ่มให้วางผ้าทับเพิ่มห้ามดึงออก",
            "หากมีสิ่งสกปรกมาก ให้ใช้น้ำเกลือปริมาณมากราดล้างแผลเพื่อไล่เศษฝุ่น ดิน หรือสิ่งแปลกปลอมออกจากแผลเบื้องต้น",
            "รีบเดินทางไปโรงพยาบาลทันที เนื่องจากแผลฉีกขาดมักต้องได้รับการเย็บแผลโดยแพทย์ และรับวัคซีนป้องกันบาดทะยัก"
        ]
    },
    "normal": {
        "title": "Normal Skin", "th_title": "ผิวหนังปกติ", "is_normal": True, "badge": "Healthy Skin",
        "findings": "ไม่พบรอยโรค การฉีกขาด หรือความผิดปกติใดๆ บนพื้นผิวหนังที่สแกน",
        "morphology": "เม็ดสีผิวมีความสม่ำเสมอ ผิวหนังมีความยืดหยุ่นและชุ่มชื้นปกติ ไม่พบสัญญาณการอักเสบ",
        "steps": [
            "ดูแลความสะอาดของผิวหนังในชีวิตประจำวัน โดยล้างทำความสะอาดด้วยสบู่สูตรอ่อนโยนที่มีค่า pH เหมาะสมกับผิว",
            "ทาโลชั่นหรือครีมบำรุงผิวเพื่อรักษาความชุ่มชื้น ป้องกันผิวหนังแห้งแตกอันเป็นสาเหตุให้เกิดบาดแผลได้ง่าย",
            "หลีกเลี่ยงการถูกแดดเผาเป็นเวลานาน และหลีกเลี่ยงการเสียดสีหรือแรงกดทับซ้ำๆ บริเวณปุ่มกระดูก"
        ]
    },
    "pressure_wound": {
        "title": "Pressure Ulcer", "th_title": "แผลกดทับ", "is_normal": False, "badge": "Stage II",
        "findings": "พบรอยโรคเนื้อเยื่อถูกทำลายเฉพาะที่จากการถูกกดทับเป็นเวลานาน สูญเสียชั้นผิวหนังบางส่วน",
        "morphology": "แผลตื้น พื้นแผลมีสีชมพูแดง (Red-pink wound bed) ไม่พบเนื้อตาย (Slough) มีอาการแดงเฉพาะที่รอบแผล",
        "steps": [
            "ทำความสะอาดแผลอย่างเบามือด้วยน้ำเกลือปราศจากเชื้อ และปิดแผลด้วยแผ่นโฟมกันกระแทก (Foam dressing) เพื่อลดแรงกด",
            "จัดตารางเวลาและทำการพลิกตะแคงตัวผู้ป่วยอย่างเคร่งครัดทุกๆ 2 ชั่วโมง เพื่อตัดวงจรไม่ให้แผลโดนกดทับซ้ำ",
            "ใช้อุปกรณ์ช่วยกระจายแรงกด เช่น ที่นอนลม และดูแลให้ผู้ป่วยได้รับสารอาหารประเภทโปรตีนอย่างเพียงพอเพื่อเร่งการซ่อมแซมผิว"
        ]
    },
    "surgical_wound": {
        "title": "Surgical Wound", "th_title": "แผลผ่าตัด", "is_normal": False, "badge": "Post-Op Care",
        "findings": "พบแผลที่เกิดจากการผ่าตัดทางการแพทย์ มีรอยเย็บหรือวัสดุเย็บแผลปรากฏชัดเจน",
        "morphology": "ขอบแผลเรียบสนิท ยึดติดกันด้วยไหมเย็บ (Sutures) หรือลวด (Staples) รอบแผลอาจบวมแดงเล็กน้อย",
        "steps": [
            "ดูแลรักษาแผลให้แห้งและสะอาดอยู่เสมอ ห้ามให้แผลโดนน้ำอย่างเด็ดขาดจนกว่าจะได้รับอนุญาตจากศัลยแพทย์เจ้าของไข้",
            "หากแพทย์สั่งให้ทำแผล ให้ใช้สำลีชุบน้ำยาฆ่าเชื้อตามที่แพทย์สั่ง เช็ดรอบๆ รอยเย็บเบาๆ จากข้างในวนออกข้างนอก",
            "เฝ้าระวังอาการติดเชื้ออย่างใกล้ชิด เช่น แผลบวมแดงมากขึ้น มีหนองไหล ปวดแผลทวีความรุนแรง หรือมีไข้สูง"
        ]
    },
    "venous_wound": {
        "title": "Venous Wound", "th_title": "แผลหลอดเลือดดำเรื้อรัง", "is_normal": False, "badge": "Vascular Ulcer",
        "findings": "พบแผลเรื้อรังบริเวณขาหรือข้อเท้า เกิดจากความดันหลอดเลือดดำสูงและไหลเวียนกลับไม่ดี",
        "morphology": "แผลมักจะตื้น ขอบแผลไม่สม่ำเสมอ พื้นแผลอาจมีน้ำเหลืองซึมออกมาก ผิวหนังรอบๆ มีสีคล้ำหรือหนาตัว",
        "steps": [
            "ล้างแผลด้วยน้ำเกลือและเลือกใช้แผ่นปิดแผลที่สามารถดูดซับน้ำเหลืองได้สูง (High-absorbency dressing) เพื่อจัดการสารคัดหลั่ง",
            "เวลานั่งหรือนอนพัก ให้ยกขาให้อยู่สูงกว่าระดับหัวใจบ่อยๆ เพื่อช่วยให้เลือดดำไหลเวียนกลับสู่หัวใจได้ดีขึ้นและลดอาการบวม",
            "ใช้ผ้าพันยืดลดบวมหรือถุงน่องกระชับกล้ามเนื้อ (Compression therapy) ตามคำสั่งของแพทย์อย่างเคร่งครัดเพื่อเพิ่มแรงดันเลือด"
        ]
    }
}

@st.cache_resource
def load_model():
    # อัปเดต: เปลี่ยนเส้นทางมาดึงไฟล์โมเดลจากไดเรกทอรีหลัก (Root) โดยตรงตามภาพ image_ec85b8.png
    return YOLO("best-8.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
    model = None

# --- ส่วนหัวจัดตรงกลางขยายเต็มหน้าจอ ---
st.markdown("""
<div style="width: 100%; display: flex; align-items: center; justify-content: center; border-bottom: 1px solid #c2c6d4; background-color: white; padding: 12px 16px; margin-bottom: 20px; border-radius: 8px;">
    <span class="material-symbols-outlined" style="color:#00478d; font-size: 24px; margin-right: 8px;">biotech</span>
    <h1 style="margin:0; font-size:22px; font-weight:700; color:#00478d; font-family:'Inter';">WoundAi</h1>
</div>
<h2 class='center-text' style='margin-top:10px; margin-bottom:2px; font-size: 24px; color:#00478d; font-weight:700;'>New Analysis</h2>
<p class='center-text' style='color:#424752; margin-bottom:20px;'>Ensure the wound is centered and well-lit for accurate diagnostic results.</p>
""", unsafe_allow_html=True)

tab_camera, tab_upload = st.tabs(["📸 กล้องถ่ายรูป", "📁 อัปโหลดไฟล์ภาพ"])
image_to_process = None

with tab_camera:
    camera_file = st.camera_input("ถ่ายภาพแผล")
    if camera_file is not None:
        image_to_process = camera_file

with tab_upload:
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = uploaded_file

if image_to_process is not None:
    st.markdown("---")
    with st.spinner("กำลังวิเคราะห์..."):
        try:
            bytes_data = image_to_process.getvalue() if hasattr(image_to_process, 'getvalue') else image_to_process.read()
            file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if model is not None and img is not None:
                results = model(img)
                res_plotted = results[0].plot()
                
                st.image(res_plotted, use_container_width=True, channels="BGR")
                
                boxes = results[0].boxes
                class_key = "normal"
                confidence_score = 0.0
                
                if len(boxes) > 0:
                    best_box_idx = np.argmax(boxes.conf.cpu().numpy())
                    class_id = int(boxes.cls[best_box_idx])
                    confidence_score = float(boxes.conf[best_box_idx]) * 100
                    # เพิ่มความปลอดภัย: รองรับกรณีกระทบความผิดพลาดของคำทำนาย (Fallback mapping)
                    raw_key = model.names[class_id].lower()
                    if raw_key in ["brun_wound", "burn_wound"]:
                        class_key = "burn_wound"
                    elif raw_key in ["laseration_wound", "laceration_wound"]:
                        class_key = "laceration_wound"
                    else:
                        class_key = raw_key
                
                if class_key not in FIRST_AID_GUIDE:
                    class_key = "normal"
                    
                guide = FIRST_AID_GUIDE[class_key]
                severity_color = "#10B981" if guide["is_normal"] else ("#F59E0B" if "stage" in guide["badge"].lower() or "chronic" in guide["badge"].lower() else "#EF4444")
                confidence_text = "ระดับความเชื่อมั่นสูง" if confidence_score > 70 else "ระดับความเชื่อมั่นปานกลาง"
                
                dash_val = 113.1 - (113.1 * confidence_score / 100)
                
                # ปรับแต่งคำอธิบายประเภทเนื้อหาให้เหมาะสมกับเคสผิวปกติ
                if guide["is_normal"]:
                    clinical_findings = guide["findings"]
                else:
                    clinical_findings = f"พบรอยโรคที่มีลักษณะเฉพาะของ <span style='font-weight: 600; color: #141d23;'>{guide['th_title']} ({guide['title']})</span> {guide['findings']}"

                step_items_html = ""
                colors_bg = ["background-color:#eff6ff; color:#1d4ed8;", "background-color:#fff7ed; color:#c2410c;", "background-color:#faf5ff; color:#6b21a8;"]
                
                for i, step_text in enumerate(guide["steps"]):
                    sc = colors_bg[i] if i < len(colors_bg) else "background-color:#f3f4f6; color:#374151;"
                    ts = f"Step {i+1}"
                    
                    step_items_html += '<div style="width: 100%; display: flex; gap: 16px; padding: 12px; border-radius: 8px; margin-bottom:10px; background-color:#ffffff; border:1px solid #f3f4f6;">'
                    step_items_html += f'<div style="height: 32px; width: 32px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; border-radius: 9999px; font-weight:700; font-size:14px; {sc}">'
                    step_items_html += f'{i+1}</div>'
                    step_items_html += f'<div><p style="margin:0; font-size:12px; font-weight:700; color:#141d23; text-transform: uppercase;">{ts}</p>'
                    step_items_html += f'<p style="margin:2px 0 0 0; font-size:14px; color:#424752;">{step_text}</p></div></div>'

                html_template = """
<div style="width: 100%; display: flex; flex-wrap: wrap; gap: 12px; margin-top: 15px; margin-bottom: 12px;">
<div style="flex: 1; min-width: 260px; border: 1px solid #E9ECEF; background: #FFFFFF; padding: 16px; border-radius: 12px; display: flex; gap: 16px; position: relative; overflow: hidden;">
<div style="background-color: {severity_color}; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; border-radius: 2px 0 0 2px;"></div>
<div style="background-color: #e0e9f2; padding: 12px; border-radius: 8px; color: #00478d; display: flex; align-items: center; justify-content: center; height: 48px; width: 48px;">
<span class="material-symbols-outlined" style="font-size: 24px;">biotech</span>
</div>
<div>
<p style="margin: 0; font-size: 12px; font-weight: 500; color: #424752;">Classification</p>
<h2 style="margin: 2px 0 0 0; font-size: 18px; font-weight: 600; color: #141d23;">{title}</h2>
<span style="display: inline-block; margin-top: 6px; background-color: #d5e3ff; color: #234778; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">{badge}</span>
</div>
</div>

<div style="flex: 1; min-width: 260px; border: 1px solid #E9ECEF; background: #FFFFFF; padding: 16px; border-radius: 12px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div style="display: flex; align-items: center; gap: 6px;">
<span class="material-symbols-outlined" style="color:#00478d; font-size: 14px;">verified</span>
<p style="margin: 0; font-size: 12px; font-weight: 500; color: #424752;">Confidence Index</p>
</div>
<span style="background-color: #dbe4ed; color: #00468c; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 9999px;">CERTIFIED AI MODEL</span>
</div>
<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 10px;">
<div>
<span style="font-size: 26px; font-weight: 700; color: #00478d; line-height: 30px;">{conf_score:.1f}%</span><br>
<span style="font-size: 12px; font-weight: 500; color: #10B981;">{conf_text}</span>
</div>
<div style="position: relative; height: 44px; width: 44px; display: flex; align-items: center; justify-content: center;">
<svg style="width: 100%; height: 100%; transform: rotate(-90deg);">
<circle cx="22" cy="22" fill="transparent" r="18" stroke="#e6eff8" stroke-width="4"></circle>
<circle cx="22" cy="22" fill="transparent" r="18" stroke-width="4" stroke="#00478d" stroke-dasharray="113.1" stroke-dashoffset="{dash_offset}"></circle>
</svg>
<span style="font-size: 9px; font-weight: 700; color: #00478d; position: absolute;">AI</span>
</div>
</div>
</div>
</div>

<section style="width: 100%; border: 1px solid #E9ECEF; background: #FFFFFF; padding: 16px; border-radius: 12px; margin-bottom:12px; font-family:'Inter';">
<h3 style="margin: 0 0 8px 0; font-size: 18px; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 8px; color: #3d5f92; text-align: center;">
<span class="material-symbols-outlined">description</span> Clinical Analysis
</h3>
<div style="display: flex; justify-content: space-between; font-size: 10px; color: #424752; border-bottom: 1px solid #c2c6d4; padding-bottom: 6px; margin-bottom: 10px;">
<span>Ref ID: #WS-9821-003</span>
<span>Status: AI Auto-Generated</span>
</div>
<div style="margin-bottom: 10px;">
<p style="margin:0; font-size: 12px; font-weight: 700; color: #00478d; text-transform: uppercase;">Key Findings (ผลการตรวจหลัก)</p>
<p style="margin: 2px 0 0 0; font-size: 14px; color: #424752;">{clinical_findings}</p>
</div>
<div style="margin-bottom: 10px;">
<p style="margin:0; font-size: 12px; font-weight: 700; color: #00478d; text-transform: uppercase;">Morphology (ลักษณะทางสัณฐานวิทยา)</p>
<p style="margin: 2px 0 0 0; font-size: 14px; color: #424752;">{morphology}</p>
</div>
</section>

<section style="width: 100%; border: 1px solid #E9ECEF; background: #FFFFFF; padding: 16px; border-radius: 12px; margin-bottom:12px; font-family:'Inter';">
<h3 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 8px; color: #3d5f92; text-align: center;">
<span class="material-symbols-outlined">assignment_turned_in</span> Next Steps (วิธีดูแลเบื้องต้น)
</h3>
<div style="width: 100%;">{steps_content}</div>
</section>

<section style="width: 100%; border: 1px solid rgba(0,93,182,0.2); background: #ecf5fe; padding: 14px; border-radius: 12px; font-family:'Inter';">
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
<div style="display: flex; align-items: center; gap: 12px;">
<div style="height: 36px; width: 36px; border-radius: 50%; background-color: rgba(0,71,141,0.1); display: flex; align-items: center; justify-content: center; color: #00478d;">
<span class="material-symbols-outlined">clinical_notes</span>
</div>
<div>
<p style="margin:0; font-size: 12px; font-weight: 700; color: #141d23;">Review Status</p>
<p style="margin:0; font-size: 10px; color: #424752;">รอยืนยันโดยแพทย์ผู้เชี่ยวชาญเพื่อความแม่นยำ</p>
</div>
</div>
<span style="background-color: #ffdad6; color: #93000a; font-size: 10px; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">PENDING REVIEW</span>
</div>
</section>
"""
                
                st.markdown(html_template.format(
                    severity_color=severity_color,
                    title=guide["title"],
                    badge=guide["badge"],
                    conf_score=confidence_score,
                    conf_text=confidence_text,
                    dash_offset=dash_val,
                    clinical_findings=clinical_findings,
                    morphology=guide["morphology"],
                    steps_content=step_items_html
                ), unsafe_allow_html=True)

            else:
                st.error("ไม่สามารถโหลดภาพหรือโมเดลสำเร็จ")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")

# --- แถบแนะนำการถ่ายภาพ (Imaging Guidelines) ด้านล่างสุด ---
st.markdown("""
    <div class="guideline-box">
        <h3 style="margin-top: 0; font-size: 16px; display: flex; align-items: center; justify-content: center; gap: 5px; color: #00478d; font-weight:600; text-align: center;">
            <span class="material-symbols-outlined">health_metrics</span> Imaging Guidelines
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; font-size: 12px; color: #424752; margin-top:8px;">
            <div><strong style="color: #00478d;">💡 LIGHTING</strong><br>Use natural or bright white clinic light.</div>
            <div><strong style="color: #00478d;">📏 DISTANCE</strong><br>Hold device 15-20cm from the wound site.</div>
            <div><strong style="color: #00478d;">🎯 STABILITY</strong><br>Hold still until focus locks for precision.</div>
            <div><strong style="color: #00478d;">✨ OBSTACLES</strong><br>Clear hair or dressing edges from view.</div>
        </div>
    </div>
""", unsafe_allow_html=True)
