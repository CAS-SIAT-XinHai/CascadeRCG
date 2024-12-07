def generation_prompt(dialogue, knowledge):
    generation_prompt = f"""
    你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面知识进行回答:
    
    ##用户的描述: {dialogue}
    ##相关知识: {knowledge}
    
    最终回答是:
    """
    return generation_prompt