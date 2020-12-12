def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("%s ===> %s" % (key, value))


if __name__ == "__main__":
    greet_me(name="python")
