from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw
import os
import io
import base64
from deeplab import DeeplabV3

# 初始化Flask应用
app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化模型
deeplab = DeeplabV3()
name_classes = ["background", "crack"]

def combine_images(original, result):
    """
    将原始图片和预测结果图片进行混合，非滑坡区域透明，滑坡区域红色半透明。
    """
    result = result.convert("RGBA")  # 转为RGBA模式
    original = original.convert("RGBA")  # 将原图转为RGBA模式

    # 获取预测结果和原始图像的数据
    result_data = result.load()
    original_data = original.load()

    # 设定滑坡区域为红色半透明
    for y in range(result.height):
        for x in range(result.width):
            if result_data[x, y][0] == 255 and result_data[x, y][1] == 255 and result_data[x, y][2] == 255:
                # 白色区域（滑坡区域）变为红色半透明
                result_data[x, y] = (255, 0, 0, 128)  # 红色，半透明
            else:
                # 非滑坡区域设置为完全透明
                result_data[x, y] = (0, 0, 0, 0)  # 完全透明

    # 将原图和结果图合并
    combined = Image.alpha_composite(original, result)

    return combined

@app.route('/')
def index():
    return render_template('index.html')  # 返回HTML页面

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected!"}), 400

    # 保存上传的文件
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    try:
        # 打开图片并进行预测
        original_image = Image.open(input_path)
        result_image = deeplab.detect_image(original_image, name_classes=name_classes)

        # 混合图片
        combined_image = combine_images(original_image, result_image)

        # 将结果转换为Base64编码
        buffer = io.BytesIO()
        combined_image.save(buffer, format="PNG")
        buffer.seek(0)
        combined_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 返回Base64编码的预测结果
        return jsonify({"message": "Prediction successful!", "combined_image": combined_image_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
