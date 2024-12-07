def single_turn_rewriting_prompt(dialogue, knowledge):
    single_turn_rewriting_prompt = f"""
    你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面知识进行回答，以提高回答的专业性和知识性。
    
    ##注意：  
    1. 以表达善意或爱意或安慰的话开头，以提供情感上的支持，回答过程中应始终保持尊重、热情、真诚、共情、积极关注的态度。 
    2. 请确保按照信息的先后顺序、逻辑关系和相关性组织文本，同时在回答中添加适当过渡句帮助读者更好理解内容之间的关系和转变。  
    3. 请你尽可能生成长文本,用中文返回,内容应该知识丰富完整且有深度
    3. 仅返回最终的回答，不要出现其他内容。
    
    ##重点：
    要求保留心理学专业术语，实验内容、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释，案例内容等。
    尽可能多得利用下面的知识。
    
    ##用户的描述：{dialogue} 
    
    ##相关知识：{knowledge} 
    最终回答是：
    """
    return single_turn_rewriting_prompt


def muti_turn_rewriting_prompt(dialogue, knowledge):
    muti_turn_rewriting_prompt = f"""
    你是一名具有丰富心理学知识且掌握多种心理咨询技巧的Counselor。请利用下面的知识对下面对话中的每一个Counselor回答进行重写。
    
    ##注意
    1.你应该对每一次Client角色的内容进行回答。
    2.保持Client内容和数量与原对话一致。 
    5.回答内容应提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。
    6.确保回答长度符合咨询场景情况，回答的内容应承接上下文，保持对话符合逻辑且流畅。
    7.回答过程中应始终保持尊重、热情、真诚、共情、积极关注的态度。
    9.用中文返回。
    
    ##重要：
    要求保留心理学专业术语，实验内容、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释，案例内容等。
    尽可能多得利用下面的知识。

    
    ##待补全对话:{dialogue}
    ##知识：{knowledge}
    ##对话格式：以Client对话开始，每一个Client后面接一个Counselor的回答，依次交替，每个角色转变应用"\n"隔开。
    
    重写对话是：
    """
    return muti_turn_rewriting_prompt

def judge_prompt(question, knowledge):
    prompt = f"""
    请判断下面知识片段内容是否对用户提问有帮助（部分内容有帮助也算有帮助）。若有帮助返回"是"，若无帮助返回“否”，仅返回是或否，不要给出理由。
    
    用户的提问：{question}
    知识片段：{knowledge}
    """
    return prompt

def extraction_prompt(test_data):
    extraction_prompt = f"""
    请对下面的问题的回答进行总结。

    ##注意：
    仅对文本进行总结，不是你去回答问题！
    要求保持心理学专业术语，实验内容、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释，案例内容等。
    问题应保持一致。

    ##问题回答对：
    {test_data}

    ##格式：
    {{"问题aaa"：“回答bbb”}}
    """
    return extraction_prompt


def query_rewriting_prompt(question):
    prompt = f"""
    你是一名心理医生，需要根据以下用户的提问来查询百科全书，根据用户的提问，你想向百科全书提出的问题是什么？
    请采用该心理现象是什么，为什么会产生该心理现象，怎么做的逻辑提问,用json的格式返回。
    用户的提问：{question}
    
    ##注意
    问题必须语义完整。
    每个问题应该独立，不依赖前一个问题。问题必须是子问题，即不包含其他问题，不超过三个问题。
    
    ##格式：
    {{"question1": "aaa"，
    "question2": "bbb"}}
    """
    return prompt