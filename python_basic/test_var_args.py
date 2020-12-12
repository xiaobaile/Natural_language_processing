def test_var_args(f_arg, *args):
    print("first normal args:", f_arg)
    for arg in args:
        print("another arg through *args:", arg)


if __name__ == "__main__":
    test_var_args("java", "python", "eggs", "test")
