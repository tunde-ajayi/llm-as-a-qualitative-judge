def bergen_processor(eval_prompt, llm_instruction):
    documents = []
    i = 1
    for i in range(1, 1000): # "max" 1k docs
        if f"Document {i}: " in llm_instruction:
            doc = llm_instruction.split(f"Document {i}: ")[1].split("\n")[0]
            documents.append(f"Document {i}: {doc}")
        else:
            break
    return eval_prompt.replace("DDD", "\n".join(documents))