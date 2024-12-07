def COT_prompt(dialogue, knowledge):
    COT_prompt = f"""
    你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面知识进行回答:
    
    ##用户的描述: {dialogue}
    ##相关知识: {knowledge}
    
    Let's think step by step!
    最终回答是:
    """
    return COT_prompt