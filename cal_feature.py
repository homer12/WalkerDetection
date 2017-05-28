# -*- coding:utf-8 -*-
import feature

rect_list = feature.cal_vertices_of_features(20,34)
print len(rect_list)
feature.write_rect_to_file(rect_list)