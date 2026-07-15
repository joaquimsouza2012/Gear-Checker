import jetson_inference
import jetson_utils
import argparse
from PIL import Image, ImageDraw, ImageFont

ALIASES = {"handbag": "backpack", "suitcase": "backpack"}
CHECKLIST = ["sports ball", "backpack", "bottle"]
CONFIDENCE_THRESHOLD = 0.2

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2")
args = parser.parse_args()

net = jetson_inference.detectNet(args.network, threshold=CONFIDENCE_THRESHOLD)
img = jetson_utils.loadImage(args.input)
detections = net.Detect(img, overlay="none")

found_items = []
keep = []
for det in detections:
    raw_name = net.GetClassDesc(det.ClassID)
    class_name = ALIASES.get(raw_name, raw_name)
    if class_name in CHECKLIST:
        found_items.append(class_name)
        keep.append((class_name, det.Confidence, det.Left, det.Top, det.Right, det.Bottom))
        print(f"Detected: {class_name} (COCO: '{raw_name}')  (confidence: {det.Confidence:.2f})")

print("\n--- Gear Check Report ---")
missing_items = []
for item in CHECKLIST:
    if item in found_items:
        print(f"[OK] {item} - packed")
    else:
        print(f"[MISSING] {item} - NOT FOUND")
        missing_items.append(item)

if missing_items:
    print(f"\nYou are missing: {', '.join(missing_items)}")
else:
    print("\nAll gear packed!")

pil = Image.open(args.input).convert("RGB")
draw = ImageDraw.Draw(pil)
W = pil.size[0]
lw = max(6, W // 400)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(30, W // 60))
except Exception:
    font = ImageFont.load_default()

for (name, conf, l, t, r, b) in keep:
    draw.rectangle([l, t, r, b], outline=(227, 192, 74), width=lw)
    label = f"{name} {conf*100:.0f}%"
    draw.text((l + lw + 6, max(0, t - W // 55)), label, fill=(227, 192, 74), font=font)

pil.save("gear_check_output.jpg", quality=90)
print("\nAnnotated image saved to: gear_check_output.jpg")
