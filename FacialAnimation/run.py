from flask import Flask, render_template, request, jsonify
from transform import sentenceCut, transToPinyin, transToMove, moveHandle
from voice import createVoice, TimeList, playAll

app = Flask(__name__)


@app.route('/')
@app.route('/question.html')
def question_page():
    return render_template('question.html')


@app.route('/answer.html')
def answer_page():
    return render_template('answer.html')


@app.route('/answer')
def answer():
    text = request.args.get('text')
    print("text:\n",text)
    text = sentenceCut(text)
    print("text:\n",text)
    # 生成音频
    createVoice([text])
    # 文本转拼音
    pinyin = transToPinyin(text)
    print("拼音pinyin:\n",pinyin)
    # 拼音转基本发音
    move = transToMove(pinyin)
    print("基本发音move:\n",move)
    # 生成动作序列
    order_list = moveHandle(move)
    print("动作序列order_list:\n",order_list)
    # 获取音频时长
    time_list = TimeList(len(order_list))
    print("音频时长order_list:\n",order_list)
    
    context = {'text': text, 'order': order_list, 'time': time_list}
    return jsonify(result=context)


@app.route('/play')
def play():
    num = request.args.get('num')
    # 播放音频
    playAll(num)
    return jsonify(result=num)


if __name__ == "__main__":
    app.run()
