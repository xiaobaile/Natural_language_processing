class Animals:

    @classmethod
    def breathe(cls):
        print("breathing")

    @classmethod
    def move(cls):
        print("moving")

    @classmethod
    def eat(cls):
        print("eating")


class Mammals(Animals):

    @classmethod
    def breastfeed(cls):
        print("feeding young")


class Cats(Mammals):
    def __init__(self, spots):
        self.spots = spots

    @staticmethod
    def catch_mouse():
        print("catching mouse")

    @staticmethod
    def left_foot_forward():
        print("left foot forward")

    @staticmethod
    def left_foot_backward():
        print("left foot backward")

    def dance(self):
        self.left_foot_forward()
        self.left_foot_backward()
        self.left_foot_forward()
        self.left_foot_backward()















