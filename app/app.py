import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import ViTForImageClassification

app = Flask(__name__)

# 设置图像保存的目录
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class_names = [
    ('Chihuahua', '吉娃娃'),
    ('Japanese_spaniel', '日本尖耳犬'),
    ('Maltese_dog', '马尔济斯'),
    ('Pekinese', '北京犬'),
    ('Shih-Tzu', '西施犬'),
    ('Blenheim_spaniel', '布伦海姆猎犬'),
    ('papillon', '蝴蝶犬'),
    ('toy_terrier', '玩具梗'),
    ('Rhodesian_ridgeback', '罗德西亚背脊犬'),
    ('Afghan_hound', '阿富汗猎犬'),
    ('basset', '巴塞特犬'),
    ('beagle', '比格犬'),
    ('bloodhound', '猎血犬'),
    ('bluetick', '蓝色猎犬'),
    ('black-and-tan_coonhound', '黑褐色浣熊猎犬'),
    ('Walker_hound', '沃克猎犬'),
    ('English_foxhound', '英国狐狸猎犬'),
    ('redbone', '红骨猎犬'),
    ('borzoi', '俄罗斯猎狼犬'),
    ('Irish_wolfhound', '爱尔兰猎狼犬'),
    ('Italian_greyhound', '意大利灵缇犬'),
    ('whippet', '惠比特犬'),
    ('Ibizan_hound', '伊比赞猎犬'),
    ('Norwegian_elkhound', '挪威猎鹿犬'),
    ('otterhound', '水獺犬'),
    ('Saluki', '萨路基猎犬'),
    ('Scottish_deerhound', '苏格兰猎鹿犬'),
    ('Weimaraner', '魏玛猎犬'),
    ('Staffordshire_bullterrier', '斯塔福郡斗牛梗'),
    ('American_Staffordshire_terrier', '美国斯塔福郡梗'),
    ('Bedlington_terrier', '贝德灵顿梗'),
    ('Border_terrier', '边境梗'),
    ('Kerry_blue_terrier', '凯里蓝梗'),
    ('Irish_terrier', '爱尔兰梗'),
    ('Norfolk_terrier', '诺福克梗'),
    ('Norwich_terrier', '诺里奇梗'),
    ('Yorkshire_terrier', '约克夏梗'),
    ('wire_haired_fox_terrier', '硬毛狐狸梗'),
    ('Lakeland_terrier', '湖区梗'),
    ('Sealyham_terrier', '西利汉梗'),
    ('Airedale', '艾尔代尔犬'),
    ('cairn', '凯恩犬'),
    ('Australian_terrier', '澳大利亚梗'),
    ('Dandie_Dinmont', '丹迪·迪蒙特梗'),
    ('Boston_bull', '波士顿斗牛犬'),
    ('miniature_schnauzer', '迷你雪纳瑞'),
    ('giant_schnauzer', '巨型雪纳瑞'),
    ('standard_schnauzer', '标准雪纳瑞'),
    ('Scotch_terrier', '苏格兰梗'),
    ('Tibetan_terrier', '西藏梗'),
    ('silky_terrier', '丝毛梗'),
    ('soft-coated_wheaten_terrier', '软毛小麦梗'),
    ('West_Highland_white_terrier', '西高地白梗'),
    ('Lhasa', '拉萨犬'),
    ('flat-coated_retriever', '平毛猎犬'),
    ('curly-coated_retriever', '卷毛猎犬'),
    ('golden_retriever', '金毛猎犬'),
    ('Labrador_retriever', '拉布拉多猎犬'),
    ('Chesapeake_Bay_retriever', '切萨皮克湾猎犬'),
    ('German_short-haired_pointer', '德国短毛指示犬'),
    ('vizsla', '匈牙利指示犬'),
    ('English_setter', '英国猎鹰犬'),
    ('Irish_setter', '爱尔兰猎鹰犬'),
    ('Gordon_setter', '戈登猎鹰犬'),
    ('Brittany_spaniel', '布列塔尼猎犬'),
    ('clumber', '克伦伯猎犬'),
    ('English_springer', '英国斯普林格犬'),
    ('Welsh_springer_spaniel', '威尔士斯普林格犬'),
    ('cocker_spaniel', '可卡犬'),
    ('Sussex_spaniel', '萨塞克斯猎犬'),
    ('Irish_water_spaniel', '爱尔兰水猎犬'),
    ('kuvasz', '库瓦兹'),
    ('schipperke', '小型荷兰牧羊犬'),
    ('groenendael', '格罗宁达尔'),
    ('malinois', '马利诺犬'),
    ('briard', '布里亚犬'),
    ('kelpie', '凯尔皮犬'),
    ('komondor', '科蒙多犬'),
    ('Old_English_sheepdog', '古老英国牧羊犬'),
    ('Shetland_sheepdog', '谢德兰牧羊犬'),
    ('collie', '可利牧羊犬'),
    ('Border_collie', '边境牧羊犬'),
    ('Bouvier_des_Flandres', '弗兰德斯牧羊犬'),
    ('Rottweiler', '罗威纳犬'),
    ('German_shepherd', '德国牧羊犬'),
    ('Doberman', '杜宾犬'),
    ('miniature_pinscher', '迷你平衡犬'),
    ('Greater_Swiss_Mountain_dog', '瑞士山地犬'),
    ('Bernese_mountain_dog', '伯恩山犬'),
    ('Appenzeller', '阿彭策尔犬'),
    ('EntleBucher', '恩特尔布赫牧羊犬'),
    ('boxer', '拳击犬'),
    ('bull_mastiff', '斗牛獒犬'),
    ('Tibetan_mastiff', '藏獒犬'),
    ('French_bulldog', '法国斗牛犬'),
    ('Great_Dane', '大丹犬'),
    ('Saint_Bernard', '圣伯纳犬'),
    ('Eskimo_dog', '爱斯基摩犬'),
    ('malamute', '马拉缪特犬'),
    ('Siberian_husky', '西伯利亚哈士奇'),
    ('affenpinscher', '阿芬犬'),
    ('basenji', '巴森吉犬'),
    ('pug', '巴哥犬'),
    ('Leonberg', '里昂堡犬'),
    ('Newfoundland', '纽芬兰犬'),
    ('Great_Pyrenees', '比利牛斯山犬'),
    ('Samoyed', '萨摩耶犬'),
    ('Pomeranian', '博美犬'),
    ('chow', '松狮犬'),
    ('keeshond', '凯士犬'),
    ('Brabancon_griffon', '布拉班松犬'),
    ('Pembroke', '彭布罗克犬(柯基)'),
    ('Cardigan', '卡迪根犬'),
    ('toy_poodle', '玩具贵宾犬'),
    ('miniature_poodle', '迷你贵宾犬'),
    ('standard_poodle', '标准贵宾犬'),
    ('Mexican_hairless', '墨西哥无毛犬'),
    ('dingo', '丁戈犬'),
    ('dhole', '野狗'),
    ('African_hunting_dog', '非洲猎犬')
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "../models/vit_finetuned_StanfordDogs_ep5"
model = ViTForImageClassification.from_pretrained(model_path)
model.to(device)
model.eval()  

trans_ = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def predict_image(image_path):
    # 打开图片并进行预处理
    image = Image.open(image_path)
    image = trans_(image).unsqueeze(0)  
    image = image.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image)
    logits = outputs.logits
    
    # 获取预测的类别索引
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html') 

# # 处理文件上传的接口
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' in request.files:
#         # 如果是文件上传
#         image = request.files['file']
#         image_path = os.path.join(UPLOAD_FOLDER, image.filename)
#         image.save(image_path)
#     else:
#         return jsonify({"error": "No file part in the request"}), 400

#     try:
#         predicted_class_idx = predict_image(image_path)
#         return jsonify({"predicted_class": class_names[predicted_class_idx]})
#     except Exception as e:
#         return jsonify({"error": str(e)})
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # 如果是文件上传
        image = request.files['file']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
    else:
        return jsonify({"error": "No file part in the request"}), 400

    try:
        predicted_class_idx = predict_image(image_path)
        
        # 获取英文和中文类别
        predicted_class_english = class_names[predicted_class_idx][0]
        predicted_class_chinese = class_names[predicted_class_idx][1]
        
        # 返回英文和中文的预测结果
        return jsonify({
            "predicted_class_english": predicted_class_english,
            "predicted_class_chinese": predicted_class_chinese
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# 你已有的上传文件夹路径
UPLOAD_FOLDER = './uploads'

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 清空上传文件夹的函数
def clear_uploads_folder():
    """清空上传文件夹中的所有文件"""
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# 新增清理上传文件夹的路由
@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    clear_uploads_folder()
    return jsonify({'message': 'Uploads folder cleared successfully'})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
