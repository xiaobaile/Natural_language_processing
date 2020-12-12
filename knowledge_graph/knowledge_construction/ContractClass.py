from utils.handle_file import read_txt_file, write_json_file
import os
from build_entity import BuildEntity


class Contract(object):
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def get_file_content(self):
        content = read_txt_file(self.input_file_path)
        return content
    
    def parse_file_content(self):
        info_entity = list()
        for line in self.get_file_content():
            entity = BuildEntity()
            line_result = line.strip().split("\t")
            entity.field_1 = line_result[0]
            entity.field_2 = line_result[1]
            entity.field_3 = line_result[2]
            entity.field_4 = line_result[3]
            entity.field_5 = line_result[4]
            info_entity.append(entity)
        write_json_file("../../data/entity_info_original.json", info_entity)
        print("finish writing entity info into json file...")
        return info_entity


def run():
    current_path = os.path.abspath(os.path.curdir)

    file_path = "../../data/201405save_data.txt"
    entity = Contract(file_path, None).parse_file_content()
    # print(entity)


if __name__ == '__main__':
    run()
    current_path = os.path.abspath(os.path.curdir)
    current_path = os.path.dirname("__file__")
    print(os.path.join(os.path.pardir, os.path.pardir, current_path))
