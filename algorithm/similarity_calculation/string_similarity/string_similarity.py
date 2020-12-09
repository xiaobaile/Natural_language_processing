import difflib


def test():
    first_sentence = "我是男生吗"
    second_sentence = "她是女生"
    result = difflib.SequenceMatcher(None, first_sentence, second_sentence).ratio()
    print(result)


if __name__ == '__main__':
    test()
