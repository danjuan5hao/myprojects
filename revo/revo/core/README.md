# REVO
## CORE
core由两个部分组成，分别是1.dst和2.policy。dst负责对话状态追踪，policy通过对话状态决定下一步的行动。
在core中，NLU被理解为一个解释器，来理解自然语言，被把他包装成为Message, 在core中流动。
### Domain
被认为是CORE所定义的世界，一切状态以及行为都应该在domain中被定义？？？也用于链接NLU和core。
#### cross domain

### DST
#### rule policy (single domain)
不处理cross domain问题。简单的通过Message中的domain, intent, slot: value 进行填充。 
#### cross domain问题
 
### Policy
#### rule policy 
不处理cross domain问题。简单的通过Message中的domain, intent, slot: value。 按序检查最近的domain中有没有slot空缺。针对空缺进行提问。 
#### action 
