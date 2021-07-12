from transform import transToPinyin, transToMove, moveHandle#文本到拼音、拼音到发音、发音到动作
from voice import createVoice, TimeList # 生成音频、获取音频时长
if __name__ == "__main__":
    #输入
    text = "在吗"
    print("文本text:\n",text)
    
    #文本到动画
    # 1.文本转拼音
    pinyin = transToPinyin(text)
    print("拼音pinyin:\n",pinyin)
    # 2.拼音转基本发音
    move = transToMove(pinyin)
    print("基本发音move:\n",move)
    # 3.生成动作序列
    order_list = moveHandle(move)
    print("动作序列order_list:\n",order_list)
    
    #文本到声音
    # 1.生成音频
    createVoice([text])
    # 2.获取音频时长
    print(len(order_list))
    time_list = TimeList()
    print("音频时长time_list:\n",order_list)
    
    #输出动画序列
    context = {'text': text, 'order': order_list, 'time': time_list}#文本、动作序列、音频时长
    print("context：\n",context)