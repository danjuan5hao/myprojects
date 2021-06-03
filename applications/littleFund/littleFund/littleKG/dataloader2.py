# -*- coding: utf-8 -*-
from pymongo import MongoClient

from py2neo import Graph,Node,Relationship
import pandas as pd 
from py2neo.matching import NodeMatcher

client = MongoClient('localhost', 27017)
fund_db = client['test']
fund_collection = fund_db["fund"]

test_graph = Graph("http://localhost:7474",username="neo4j",  password="1234")
test_graph.delete_all()

test_graph.schema.create_uniqueness_constraint('Manager', 'manager_id')
test_graph.schema.create_uniqueness_constraint('Fund', 'fund_id')
test_graph.schema.create_uniqueness_constraint('Stock', 'stock_id')

matcher = NodeMatcher(test_graph)
l = fund_collection.find()

for idx, sample in enumerate(l):
    fund_name = sample["fund_name"]
    fund_id = sample['fund_id']

    fund_node_rst = matcher.match("Fund",fund_id=fund_id)

    if len(fund_node_rst):
        fund_node = fund_node_rst.first()
    else:
        fund_node = Node("Fund", fund_name=fund_name, fund_id=fund_id)
        test_graph.create(fund_node)

    stocks = sample["stocks"]
    if stocks:
        for stock in stocks:
            stock_name = stock["stack_name"]
            stock_id = stock["stack_id"]

            stock_node_rst = matcher.match("Stock",stock_id=stock_id)

            if len(stock_node_rst):
                stock_node = stock_node_rst.first()
            else:
                stock_node = Node("Stock", stock_id=stock_id, stock_name=stock_name)
                test_graph.create(stock_node)

            rel_to = Relationship(fund_node, "contain", stock_node)
            rel_from = Relationship(stock_node, "belong", fund_node)

            s = fund_node | stock_node | rel_to | rel_from
            test_graph.create(s)








       