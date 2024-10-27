from flask import Flask, send_from_directory

app = Flask(__name__)

# 设置公共目录（public）作为静态文件目录
@app.route('/')
def index():
    # 默认访问 index.html
    return send_from_directory('public', 'index.html')

# 为其他 HTML 文件设置路由
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

if __name__ == '__main__':
    # 运行 Flask 服务器
    app.run(host="0.0.0.0", port=8089)