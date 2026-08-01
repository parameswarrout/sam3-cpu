import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Please run: pip install gradio")
    import sys
    sys.exit(1)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Default paths and global model state
DEFAULT_CHECKPOINT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "sam3", "sam3.pt")
)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None
current_device = DEFAULT_DEVICE
current_checkpoint = DEFAULT_CHECKPOINT_PATH
click_history = [] # Global storage for interactive points: (x_norm, y_norm, label, px, py)

def load_model(checkpoint_path, device_choice):
    """Load or reload the SAM 3 model dynamically from UI."""
    global model, processor, current_device, current_checkpoint

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return (
            f"❌ Checkpoint file not found at: {checkpoint_path}",
            "🔴 Model Not Loaded",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    try:
        device = "cuda" if (device_choice == "CUDA" and torch.cuda.is_available()) else "cpu"
        print(f"[UI Request] Loading SAM 3 from {checkpoint_path} on {device}...")
        
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
        )
        processor = Sam3Processor(model, device=device)
        current_device = device
        current_checkpoint = checkpoint_path
        
        status_msg = f"✅ Model loaded successfully on {device.upper()} from {os.path.basename(checkpoint_path)}!"
        badge_msg = f"🟢 READY ({device.upper()})"
        return (
            status_msg,
            badge_msg,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return (
            f"❌ Failed to load model: {str(e)}",
            "🔴 Loading Failed",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

def overlay_masks(image, masks, boxes=None, labels=None):
    """Overlay masks and bounding boxes directly onto original image with zero white borders."""
    img_np = np.array(image.convert("RGB")).copy()
    h, w, _ = img_np.shape

    colors = [
        [230, 50, 50],   # Red
        [50, 180, 80],   # Green
        [50, 100, 230],  # Blue
        [230, 180, 30],  # Yellow
        [200, 50, 200],  # Purple
        [30, 200, 200],  # Cyan
    ]

    if masks is not None:
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        for idx, mask in enumerate(masks):
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > 0.5
            if mask_bool.shape[:2] != (h, w):
                mask_pil = Image.fromarray((mask_bool * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                mask_bool = np.array(mask_pil) > 128

            color = np.array(colors[idx % len(colors)], dtype=np.uint8)
            alpha = 0.45
            img_np[mask_bool] = (
                img_np[mask_bool] * (1 - alpha) + color * alpha
            ).astype(np.uint8)

    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if boxes is not None:
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = tuple(colors[idx % len(colors)])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            if labels is not None and idx < len(labels):
                label_text = str(labels[idx])
                text_bbox = draw.textbbox((x1, max(y1 - 18, 0)), label_text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1 + 2, max(y1 - 18, 0)), label_text, fill="white", font=font)

    return img_pil

def draw_point_markers(image, history):
    """Draw green (+) and red (-) dots for clicked points on image."""
    img_pil = image.copy()
    draw = ImageDraw.Draw(img_pil)
    radius = 6
    for x_norm, y_norm, label, px, py in history:
        color = (50, 220, 80) if label == 1 else (240, 50, 50)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=(255, 255, 255), width=2)
    return img_pil

def segment_text(image, text_prompt, confidence):
    global processor
    if processor is None:
        return None, "⚠️ Model is not loaded. Please click '🚀 Load / Reload SAM 3 Model' first."
    if image is None:
        return None, "Please upload or select an image."
    if not text_prompt or text_prompt.strip() == "":
        return None, "Please enter a text prompt (e.g. 'person', 'car', 'dog')."

    processor.set_confidence_threshold(confidence)
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=text_prompt, state=state)

    masks = output.get("masks", None)
    boxes = output.get("boxes", None)

    if masks is None or len(masks) == 0:
        return np.array(image), f"No objects found matching '{text_prompt}' with confidence >= {confidence:.2f}. Lower the slider to 0.05 - 0.10!"

    num_found = len(masks)
    result_img = overlay_masks(image, masks, boxes, labels=[text_prompt]*num_found)
    return result_img, f"🎉 Success! Found {num_found} object(s) matching '{text_prompt}'!"

def handle_click(image, click_mode, evt: gr.SelectData):
    global processor, click_history
    if processor is None:
        return None, "⚠️ Model is not loaded. Click '🚀 Load / Reload SAM 3 Model' first."
    if image is None:
        return None, "Please upload an image first."

    px, py = evt.index
    width, height = image.size

    x_norm = px / width
    y_norm = py / height
    label = 1 if click_mode == "🟢 Positive Click (+)" else 0

    click_history.append((x_norm, y_norm, label, px, py))

    points = [[p[0], p[1]] for p in click_history]
    labels = [p[2] for p in click_history]

    state = processor.set_image(image)
    output = processor.add_point_prompt(points=points, labels=labels, state=state)

    masks = output.get("masks", None)
    boxes = output.get("boxes", None)

    result_img = overlay_masks(image, masks, boxes)
    result_img = draw_point_markers(result_img, click_history)

    pos_count = sum(1 for p in click_history if p[2] == 1)
    neg_count = sum(1 for p in click_history if p[2] == 0)
    info_msg = f"🎯 Processed {len(click_history)} point(s) ({pos_count} positive 🟢, {neg_count} negative 🔴). Segmented {len(masks) if masks is not None else 0} region(s)!"
    return result_img, info_msg

def clear_points(image):
    global click_history
    click_history = []
    if image is None:
        return None, "Cleared all clicked points."
    return np.array(image), "Cleared all clicked points."

def undo_point(image):
    global click_history, processor
    if len(click_history) > 0:
        click_history.pop()
    if image is None:
        return None, "Undid last point."
    if len(click_history) == 0 or processor is None:
        return np.array(image), "No points remaining."

    points = [[p[0], p[1]] for p in click_history]
    labels = [p[2] for p in click_history]

    state = processor.set_image(image)
    output = processor.add_point_prompt(points=points, labels=labels, state=state)

    masks = output.get("masks", None)
    boxes = output.get("boxes", None)

    result_img = overlay_masks(image, masks, boxes)
    result_img = draw_point_markers(result_img, click_history)
    return result_img, f"Undid point. {len(click_history)} point(s) remaining."

# Custom CSS for rich dark mode glassmorphism theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%) !important;
    color: #f3f4f6 !important;
}

.main-card {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

.header-title {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    margin-bottom: 4px !important;
}

.sub-title {
    color: #94a3b8 !important;
    font-size: 1.05rem !important;
}

.status-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.load-btn {
    background: linear-gradient(90deg, #0284c7 0%, #6366f1 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
}

.segment-btn {
    background: linear-gradient(90deg, #059669 0%, #0d9488 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.4) !important;
}
"""

with gr.Blocks(title="SAM 3 AI Vision Suite") as demo:
    # Header Banner
    with gr.Row(elem_classes=["main-card"]):
        with gr.Column(scale=3):
            gr.Markdown("<div class='header-title'>⚡ SAM 3 Interactive Vision Suite</div>")
            gr.Markdown("<div class='sub-title'>Next-Generation Real-Time Text & Interactive Multi-Point Object Segmentation Engine</div>")
        with gr.Column(scale=1, min_width=200):
            model_badge = gr.Markdown("<div class='status-badge'>🟢 READY (CPU)</div>")

    # Model Loader Section UI
    with gr.Accordion("⚙️ Model Configuration & Loading Control", open=True, elem_classes=["main-card"]):
        with gr.Row():
            ckpt_input = gr.Textbox(
                value=DEFAULT_CHECKPOINT_PATH,
                label="Local Model Checkpoint (.pt)",
                placeholder="Path to sam3.pt checkpoint file...",
                scale=3
            )
            device_radio = gr.Radio(
                choices=["CPU", "CUDA"],
                value="CUDA" if torch.cuda.is_available() else "CPU",
                label="Execution Device",
                scale=1
            )
            load_btn = gr.Button("🚀 Load / Reload SAM 3 Model", elem_classes=["load-btn"], scale=1)

        load_status = gr.Textbox(
            value=f"Model initialized on {DEFAULT_DEVICE.upper()} from {os.path.basename(DEFAULT_CHECKPOINT_PATH)}",
            label="Model Status Log",
            interactive=False
        )

    # Main Tabs Section
    with gr.Tabs():
        # TAB 1: Text Prompt Segmentation
        with gr.TabItem("📝 Text Prompt Grounding"):
            with gr.Row(elem_classes=["main-card"]):
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="1. Upload Image")
                    text_input = gr.Textbox(
                        label="2. Text Prompt",
                        placeholder="Type objects e.g. 'person', 'dog', 'cup', 'car'...",
                        value="person"
                    )
                    conf_slider = gr.Slider(
                        minimum=0.01, maximum=0.70, value=0.10, step=0.01,
                        label="3. Confidence Threshold (Default 0.10)"
                    )
                    btn_segment = gr.Button("🔍 Segment Objects", elem_classes=["segment-btn"])

                with gr.Column(scale=1):
                    output_img = gr.Image(type="numpy", label="Segmentation Result")
                    status_text = gr.Textbox(label="Detection Results Summary", interactive=False)

        # TAB 2: Interactive Point Selection
        with gr.TabItem("🎯 Interactive Multi-Point Selection"):
            with gr.Row(elem_classes=["main-card"]):
                with gr.Column(scale=1):
                    click_mode_radio = gr.Radio(
                        choices=["🟢 Positive Click (+)", "🔴 Negative Click (-)"],
                        value="🟢 Positive Click (+)",
                        label="Click Mode (Add Object or Exclude Background)"
                    )
                    click_img = gr.Image(type="pil", label="Click Anywhere on Image to Add Positive/Negative Points")
                    with gr.Row():
                        undo_btn = gr.Button("↩️ Undo Last Point")
                        clear_btn = gr.Button("🗑️ Clear All Points")

                with gr.Column(scale=1):
                    click_output = gr.Image(type="numpy", label="Interactive Point Segmentation Result")
                    click_status = gr.Textbox(label="Click Selection Summary", interactive=False)

    # Auto-load model on startup
    demo.load(
        fn=load_model,
        inputs=[ckpt_input, device_radio],
        outputs=[load_status, model_badge, btn_segment, click_img]
    )

    # UI Event Bindings
    load_btn.click(
        fn=load_model,
        inputs=[ckpt_input, device_radio],
        outputs=[load_status, model_badge, btn_segment, click_img]
    )

    btn_segment.click(
        fn=segment_text,
        inputs=[input_img, text_input, conf_slider],
        outputs=[output_img, status_text]
    )

    click_img.select(
        fn=handle_click,
        inputs=[click_img, click_mode_radio],
        outputs=[click_output, click_status]
    )

    clear_btn.click(
        fn=clear_points,
        inputs=[click_img],
        outputs=[click_output, click_status]
    )

    undo_btn.click(
        fn=undo_point,
        inputs=[click_img],
        outputs=[click_output, click_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)
