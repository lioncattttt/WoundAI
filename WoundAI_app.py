import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# --- การตั้งค่าหน้าเว็บและการแสดงผลแบบเต็มหน้าจอ (Wide Layout) ---
st.set_page_config(page_title="WoundScan Pro - Analysis", layout="wide")

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

# --- คลังข้อมูลวิธีปฐมพยาบาล 10 คลาส ---
FIRST_AID_GUIDE = {
    "abrasion_wound": {
        "title": "Abrasion Wound", "th_title": "แผลถลอก", "is_normal": False, "badge": "Surface Injury",
        "findings": "พบรอยโรคที่มีลักษณะเป็นแผลตื้นบริเวณผิวหนังกำพร้า มีการถลอกจากการครูดหรือเสียดสี",
        "morphology": "มีเลือดออกซึมเล็กน้อย ขอบแผลไม่เรียบ พื้นแผลมีสีแดงเรื่อ อาจมีเศษดินหรือสิ่งสกปรกปนเปื้อนเล็กน้อย",
        "steps": ["Clean with sterile saline; remove any visible debris gently.", "Apply a thin layer of topical antiseptic or antibiotic ointment.", "Cover with a non-stick sterile pad if the area is prone to friction."]
    },
    "bruises_wound": {
        "title": "Bruises Wound", "th_title": "แผลฟกช้ำ", "is_normal": False, "badge": "Hematoma",
        "findings": "พบการสะสมของเลือดใต้ชั้นผิวหนังจากการกระแทก โดยไม่มีรอยฉีกขาดของผิวหนังภายนอก",
        "morphology": "ผิวหนังมีสีม่วง คล้ำ เขียว หรือเหลืองตามอายุของรอยช้ำ มีอาการบวมและเจ็บเสียวเมื่อกด",
        "steps": ["Apply cold compress for 15-20 minutes during the first 48 hours.", "Switch to warm compress after 48 hours to promote blood circulation.", "Elevate the affected limb and rest to reduce localized swelling."]
    },
    "burn_wound": {  
        "title": "Burn Wound", "th_title": "แผลไฟไหม้ / น้ำร้อนลวก", "is_normal": False, "badge": "Thermal Injury",
        "findings": "พบการทำลายของเนื้อเยื่อผิวหนังจากความร้อน สารเคมี หรือกระแสไฟฟ้า",
        "morphology": "ผิวหนังแดงจัด บวม มีตุ่มน้ำพอง (Blisters) หรือผิวหนังลอกหลุด มีความรู้สึกปวดแสบปวดร้อนรุนแรง",
        "steps": ["Cool the burn under gentle, running room-temperature water for 10-20 mins.", "Do NOT apply ice, toothpaste, or butter. Keep the wound sterile.", "Cover loosely with a clean plastic wrap or sterile non-adherent gauze."]
    },
    "brun_wound": {  # ป้องกันกรณี Model ทำนายสะกดผิด
        "title": "Burn Wound", "th_title": "แผลไฟไหม้ / น้ำร้อนลวก", "is_normal": False, "badge": "Thermal Injury",
        "findings": "พบการทำลายของเนื้อเยื่อผิวหนังจากความร้อน สารเคมี หรือกระแสไฟฟ้า",
        "morphology": "ผิวหนังแดงจัด บวม มีตุ่มน้ำพอง (Blisters) หรือผิวหนังลอกหลุด มีความรู้สึกปวดแสบปวดร้อนรุนแรง",
        "steps": ["Cool the burn under gentle, running room-temperature water for 10-20 mins.", "Do NOT apply ice, toothpaste, or butter. Keep the wound sterile.", "Cover loosely with a clean plastic wrap or sterile non-adherent gauze."]
    },
    "cut_wound": {
        "title": "Cut Wound", "th_title": "แผลถูกบาด / แผลของมีคม", "is_normal": False, "badge": "Incised Wound",
        "findings": "พบรอยแยกของผิวหนังเป็นทางยาวขอบเรียบ เกิดจากของมีคมบาด",
        "morphology": "ปากแผลเรียบชิดติดกันหรือแยกออกเล็กน้อย มีเลือดไหลค่อนข้างมาก ความลึกขึ้นอยู่กับแรงกด",
        "steps": ["Apply direct pressure with a clean cloth to stop bleeding immediately.", "Clean the surrounding area with soap and water, flush wound with saline.", "Apply antibiotic ointment and secure with a sterile adhesive bandage."]
    },
    "diabetic_wound": {
        "title": "Diabetic Wound", "th_title": "แผลเบาหวาน", "is_normal": False, "badge": "Chronic Ulcer",
        "findings": "พบแผลเรื้อรังบริเวณส่วนปลาย (มักเป็นที่เท้า) ในผู้ป่วยที่มีประวัติโรคเบาหวาน",
        "morphology": "ขอบแผลหนาและแข็ง (Callus) พื้นแผลมักลึก การไหลเวียนเลือดส่วนปลายลดลง อาจมีความรู้สึกชา",
        "steps": ["Clean gently with sterile saline; avoid harsh chemicals like alcohol inside wound.", "Apply specialized diabetic moisture-balancing dressing; change daily.", "Strictly avoid weight-bearing on the wound area and monitor blood sugar."]
    },
    "laceration_wound": { 
        "title": "Laceration Wound", "th_title": "แผลฉีกขาดฉกรรจ์", "is_normal": False, "badge": "Trauma Injury",
        "findings": "พบแผลฉีกขาดขอบไม่เรียบขรุขระจากการถูกของแข็งกระแทกหรือฉีกกระชาก",
        "morphology": "เนื้อเยื่อมีการฉีกขาดรุ่งริ่ง แผลมักจะลึกและกว้าง มีเลือดออกปานกลางถึงมาก มักมีสิ่งปนเปื้อน",
        "steps": ["Control bleeding with continuous direct pressure using sterile gauze.", "Irrigate thoroughly with clean running water or saline to flush out dirt.", "Seek immediate medical attention for closure/suturing and tetanus toxoid."]
    },
    "laseration_wound": {  # ป้องกันกรณี Model ทำนายสะกดผิด
        "title": "Laceration Wound", "th_title": "แผลฉีกขาดฉกรรจ์", "is_normal": False, "badge": "Trauma Injury",
        "findings": "พบแผลฉีกขาดขอบไม่เรียบขรุขระจากการถูกของแข็งกระแทกหรือฉีกกระชาก",
        "morphology": "เนื้อเยื่อมีการฉีกขาดรุ่งริ่ง แผลมักจะลึกและกว้าง มีเลือดออกปานกลางถึงมาก มักมีสิ่งปนเปื้อน",
        "steps": ["Control bleeding with continuous direct pressure using sterile gauze.", "Irrigate thoroughly with clean running water or saline to flush out dirt.", "Seek immediate medical attention for closure/suturing and tetanus toxoid."]
    },
    "normal": {
        "title": "Normal Skin", "th_title": "ผิวหนังปกติ", "is_normal": True, "badge": "Healthy Skin",
        "findings": "ไม่พบรอยโรค การฉีกขาด หรือความผิดปกติใดๆ บนพื้นผิวหนังที่สแกน",
        "morphology": "เม็ดสีผิวมีความสม่ำเสมอ ผิวหนังมีความยืดหยุ่นและชุ่มชื้นปกติ ไม่พบสัญญาณการอักเสบ",
        "steps": ["Maintain baseline skin hygiene by washing with mild, soap-free cleanser.", "Keep skin adequately hydrated with appropriate medical-grade moisturizers.", "Avoid repetitive friction or shear forces on prominent bony areas."]
    },
    "pressure_wound": {
        "title": "Pressure Ulcer", "th_title": "แผลกดทับ", "is_normal": False, "badge": "Stage II",
        "findings": "พบรอยโรคเนื้อเยื่อถูกทำลายเฉพาะที่จากการถูกกดทับเป็นเวลานาน สูญเสียชั้นผิวหนังบางส่วน",
        "morphology": "แผลตื้น พื้นแผลมีสีชมพูแดง (Red-pink wound bed) ไม่พบเนื้อตาย (Slough) มีอาการแดงเฉพาะที่รอบแผล",
        "steps": ["Clean with sterile saline; apply non-adherent foam dressing to retain moisture.", "Implement a strict 2-hour repositioning schedule to eliminate pressure.", "Utilize pressure-relieving mattress overlays and optimize protein intake."]
    },
    "surgical_wound": {
        "title": "Surgical Wound", "th_title": "แผลผ่าตัด", "is_normal": False, "badge": "Post-Op Care",
        "findings": "พบแผลที่เกิดจากการผ่าตัดทางการแพทย์ มีรอยเย็บหรือวัสดุเย็บแผลปรากฏชัดเจน",
        "morphology": "ขอบแผลเรียบสนิท ยึดติดกันด้วยไหมเย็บ (Sutures) หรือลวด (Staples) รอบแผลอาจบวมแดงเล็กน้อย",
        "steps": ["Keep the wound completely dry as instructed by your primary surgeon.", "Clean the surrounding skin gently with prescribed antiseptic solutions.", "Monitor closely for signs of infection like pus, increased pain, or fever."]
    },
    "venous_wound": {
        "title": "Venous Wound", "th_title": "แผลหลอดเลือดดำเรื้อรัง", "is_normal": False, "badge": "Vascular Ulcer",
        "findings": "พบแผลเรื้อรังบริเวณขาหรือข้อเท้า เกิดจากความดันหลอดเลือดดำสูงและไหลเวียนกลับไม่ดี",
        "morphology": "แผลมักจะตื้น ขอบแผลไม่สม่ำเสมอ พื้นแผลอาจมีน้ำเหลืองซึมออกมาก ผิวหนังรอบๆ มีสีคล้ำหรือหนาตัว",
        "steps": ["Cleanse the wound bed gently with saline; manage heavily exudating fluid.", "Elevate the legs above heart level whenever resting to reduce venous pooling.", "Apply multi-layer compression bandages strictly under medical supervision."]
    }
}

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
    model = None

# --- ส่วนหัวจัดตรงกลางขยายเต็มหน้าจอ ---
st.markdown("""
<div style="width: 100%; display: flex; align-items: center; justify-content: center; border-bottom: 1px solid #c2c6d4; background-color: white; padding: 12px 16px; margin-bottom: 20px; border-radius: 8px;">
    <span class="material-symbols-outlined" style="color:#00478d; font-size: 24px; margin-right: 8px;">biotech</span>
    <h1 style="margin:0; font-size:22px; font-weight:700; color:#00478d; font-family:'Inter';">WoundScan Pro</h1>
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
                
                # แสดงภาพผลลัพธ์จาก AI ให้เต็มตาม Container ของ Wide Layout พร้อมแก้สี BGR
                st.image(res_plotted, use_container_width=True, channels="BGR")
                
                boxes = results[0].boxes
                class_key = "normal"
                confidence_score = 0.0
                
                if len(boxes) > 0:
                    best_box_idx = np.argmax(boxes.conf.cpu().numpy())
                    class_id = int(boxes.cls[best_box_idx])
                    confidence_score = float(boxes.conf[best_box_idx]) * 100
                    class_key = model.names[class_id].lower()
                
                if class_key not in FIRST_AID_GUIDE:
                    class_key = "normal"
                    
                guide = FIRST_AID_GUIDE[class_key]
                severity_color = "#10B981" if guide["is_normal"] else ("#F59E0B" if "stage" in guide["badge"].lower() or "chronic" in guide["badge"].lower() else "#EF4444")
                confidence_text = "ระดับความเชื่อมั่นสูง" if confidence_score > 70 else "ระดับความเชื่อมั่นปานกลาง"
                
                # แยกคำนวณ SVG offset
                dash_val = 113.1 - (113.1 * confidence_score / 100)
                
                # ปรับแต่งหัวข้อย่อยให้ยืดหยุ่นตามลำดับขั้นตอนจริง ไม่ Hardcode
                step_items_html = ""
                icons = ["cleaning_services", "healing", "update"]
                colors_bg = ["background-color:#eff6ff; color:#1d4ed8;", "background-color:#fff7ed; color:#c2410c;", "background-color:#faf5ff; color:#6b21a8;"]
                titles_step = ["Step 1: Primary Action", "Step 2: Protection", "Step 3: Monitoring"]
                
                for i, step_text in enumerate(guide["steps"]):
                    sc = colors_bg[i] if i < len(colors_bg) else "background-color:#f3f4f6; color:#374151;"
                    ts = titles_step[i] if i < len(titles_step) else f"Step {i+1}"
                    ic = icons[i] if i < len(icons) else "add_task"
                    
                    step_items_html += '<div style="width: 100%; display: flex; gap: 16px; padding: 12px; border-radius: 8px; margin-bottom:10px; background-color:#ffffff; border:1px solid #f3f4f6;">'
                    step_items_html += f'<div style="height: 32px; width: 32px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; border-radius: 9999px; {sc}">'
                    step_items_html += f'<span class="material-symbols-outlined" style="font-size:18px;">{ic}</span></div>'
                    step_items_html += f'<div><p style="margin:0; font-size:12px; font-weight:700; color:#141d23;">{ts}</p>'
                    step_items_html += f'<p style="margin:2px 0 0 0; font-size:14px; color:#424752;">{step_text}</p></div></div>'

                # เพิ่ม ปีกกาซ้อนสองชั้น {{ }} ใน CSS/SVG เพื่อหลบเลี่ยง String format ของ Python บั๊กหลักถูกแก้ที่นี่ครับ
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
<p style="margin: 2px 0 0 0; font-size: 14px; color: #424752;">พบรอยโรคที่มีลักษณะเฉพาะของ <span style="font-weight: 600; color: #141d23;">{th_title} ({title})</span> {findings}</p>
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
                
                # ส่งค่าเข้าไปแทนที่ในจุดปลอดภัยโดยใช้ฟังก์ชัน .format()
                st.markdown(html_template.format(
                    severity_color=severity_color,
                    title=guide["title"],
                    badge=guide["badge"],
                    conf_score=confidence_score,
                    conf_text=confidence_text,
                    dash_offset=dash_val,
                    th_title=guide["th_title"],
                    findings=guide["findings"],
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
