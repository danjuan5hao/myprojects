# -*- coding: utf-8 -*-
from py2neo import Graph,Node,Relationship
import pandas as pd 
from py2neo.matching import NodeMatcher


from py2neo.ogm import GraphObject, Property, RelatedTo, RelatedFrom, Label
# class Person(GraphObject):
#     # 定义主键
#     __primarykey__ = 'name'
#     # 定义类的属性
#     name=Property()
#     age=Property()
#     # 定义类的标签
#     student=Label()
#     # 定义Person指向的关系
#     knows=RelatedTo('Person','KNOWS')
#     # 定义指向Person的关系
#     known=RelatedFrom('Person','KNOWN')
# ————————————————
# 版权声明：本文为CSDN博主「Vic·Tory」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/theVicTory/article/details/100527171

# class Manager(GraphObject):
#     __primarykey__ = 'manager_id'
#     manager_name = Property()
#     manager_id = Property()
#     run=RelatedTo('Fund','run')





# class Fund(GraphObject):
#     __primarykey__ = 'fund_id'
#     fund_name = Property()
#     fund_id = Property()
#     runby=RelatedTo('Manager','runby')






if __name__ == "__main__":
    manager_file_path = "applications/littleFund/data/littleKG/manager_20210601.csv"

    test_graph = Graph("http://localhost:7474", auth=("neo4j", '1234') )

    # test_graph.delete_all()
    test_graph.schema.create_uniqueness_constraint('Manager', 'manager_id')
    test_graph.schema.create_uniqueness_constraint('Fund', 'fund_id')

    matcher = NodeMatcher(test_graph)
    # manager_df = pd.read_csv(manager_path, dtype={'fund_id': str} )


    manager_df = pd.read_csv(manager_file_path, dtype={"fund_id": str, "manager_id":str})
    manager_df['fund_id'] = manager_df['fund_id'].apply(lambda x: str(x).lstrip())
    manager_df['manager_id'] = manager_df['manager_id'].apply(lambda x: str(x).lstrip())
    for idx, row in manager_df.iterrows():
        manager_id,	manager_name,company_id,company_name,avatar,start_day,scale,best_reward,description,fund_id,fund_name = row
        # print(manager_id)
        
        manager_node_rst = matcher.match("Manager",manager_id=manager_id)

        if len(manager_node_rst):
            manager_node = manager_node_rst.first()
        else:
            manager_node = Node("Manager", manager_id=manager_id, manager_name=manager_name)
            test_graph.create(manager_node)
        

        fund_node_rst = matcher.match("Fund",fund_id=fund_id)

        if len(fund_node_rst):
            fund_node = fund_node_rst.first()
        else:
            fund_node = Node("Fund", fund_id=fund_id, fund_name=fund_name)
            test_graph.create(fund_node)
        
        rel_to = Relationship(manager_node, "run", fund_node)
        rel_from = Relationship(fund_node, "runby", manager_node)

        s = fund_node | manager_node | rel_to | rel_from
        test_graph.create(s) 
        test_graph.push(s)
        # exit()
        





