import pypinyin
#通过将文本转换成动作序列

def sentenceCut(sentence):
    loc = sentence.find('。', 100)
    if loc == -1:
        return sentence
    else:
        sent = sentence[0:loc]
        sent += '。'
        return sent


def transToPinyin(sentences):
    s = ""
    for index, value in enumerate(pypinyin.pinyin(sentences, style=pypinyin.NORMAL)):
        s += "".join(value)
        s += " "
    s = s[:-1]
    return [s]


def transToMove(pinyin_list):
    # print("拼音：", pinyin_list)
    move_list = []
    for pinyin in pinyin_list:
        p_list = pinyin.split(" ")
        m_list = ""
        for p in p_list:
            m = PtoM(p)
            m_list += m
            m_list += " "
        # print(m_list)
        m_list = m_list[:-1]
        move_list.append(m_list)
    # print("动作：", move_list)
    return move_list


def PtoM(pinyin):
    # print("拼音：", pinyin)
    move = ""
    rest = ""
    if pinyin[0] == 'b' or pinyin[0] == 'p' or pinyin[0] == 'm' or pinyin[0] == 'f':
        move += "b"
        rest = pinyin[1:]
    elif pinyin[0] == 'd' or pinyin[0] == 't' or pinyin[0] == 'n' or pinyin[0] == 'l' or pinyin[0] == 'g' or pinyin[
        0] == 'k' or pinyin[0] == 'h' or pinyin[0] == 'j' or pinyin[0] == 'q' or pinyin[0] == 'x' or pinyin[0] == 'z' or \
            pinyin[0] == 'c' or pinyin[0] == 's' or pinyin[0] == 'r':
        move += "d"
        if pinyin[1] == 'h':
            rest = pinyin[2:]
        else:
            rest = pinyin[1:]
    elif pinyin[0] == 'y':
        move += "e"
        rest = pinyin[1:]
    elif pinyin[0] == 'w':
        move += "u"
        rest = pinyin[1:]
    else:
        if is_number(pinyin):
            move += numberHandle(pinyin)[0]

    if rest == "a" or rest == "ia" or rest == "ua" or rest == "ai" or rest == "uai" or rest == "ao" or rest == "iao" or rest == "an" or rest == "ian" or rest == "uan" or rest == "van" or rest == "ang" or rest == "iang" or rest == "uang":
        move += "a"
    elif rest == "o" or rest == "ou" or rest == "ong" or rest == "iong" or rest == "u" or rest == "uo" or rest == "v" or rest == "ve" or rest == "iou" or rest == "un" or rest == "ui" or rest == "iu":
        move += "u"
    elif rest == "e" or rest == "i" or rest == "ie" or rest == "er" or rest == "ei" or rest == "uei" or rest == "en" or rest == "in" or rest == "uen" or rest == "eng" or rest == "ing" or rest == "ueng" or rest == "ue":
        move += "e"

    # print("动作：", move)
    return move


def MtoO(p):
    if p == 'b':
        return 0
    elif p == 'd':
        return 1
    elif p == 'a':
        return 2
    elif p == 'e':
        return 3
    elif p == 'u':
        return 4


def moveHandle(move_list):
    order_list = []
    for move in move_list:
        Pmove = move.split(' ')
        # print(Pmove)
        orders = []
        for p2 in Pmove:
            for p1 in p2:
                order = MtoO(p1)
                orders.append(order)
        order_list.append(orders)
    # print(order_list)
    return order_list


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False


def numberHandle(num):
    text = [tr_digit_to_zn(int(num))]
    pinyin = transToPinyin(text)
    move = transToMove(pinyin)
    return move


# 阿拉伯数字转换成汉语
def tr_digit_to_zn(digit):
    # 940,2400,0452
    digit = str(digit)
    length = len(digit)
    digit = digit[::-1]
    split = []
    sp_nums = range(0, length, 4)
    for i in sp_nums:
        split.append(digit[i: i + 4][::-1].zfill(4))
    # print(split)
    d_digit_to_zn = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
    }
    res_zn_list = []
    split_count = len(split)
    for i, e in enumerate(split):
        zn = ''
        for j, each in enumerate(e):
            if each == '0':
                if j == 0 and i == split_count - 1:
                    pass
                elif e[j - 1] == '0':
                    pass
                elif e[j:].strip('0'):
                    zn += '零'
            else:
                zn += d_digit_to_zn[int(each)] + {0: '千', 1: '百', 2: '十', 3: ''}[j]
        zn = zn + {0: '', 1: '万', 2: '亿'}[i]
        res_zn_list.append(zn)
    res_zn_list.reverse()
    res_zn = ''.join(res_zn_list)
    # print(res_zn)

    res_zn = [e for e in res_zn]
    for i, e in enumerate(res_zn):
        if e in '百千':
            try:
                if res_zn[i - 1] == '二':
                    res_zn[i - 1] = '两'
            except:
                pass
    res_zn = ''.join(res_zn)

    if res_zn.startswith('一十'):
        res_zn = res_zn[1:]

    if res_zn.startswith('二'):
        if len(res_zn) == 1:
            res_zn = '两'
        elif res_zn[1] in ['万', '亿']:
            res_zn = '两' + res_zn[1:]

    return res_zn
